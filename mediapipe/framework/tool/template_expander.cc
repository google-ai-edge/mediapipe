// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/tool/template_expander.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/port/numbers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/calculator_graph_template.pb.h"
#include "mediapipe/framework/tool/proto_util_lite.h"

namespace mediapipe {

namespace tool {

using mediapipe::proto_ns::MessageLite;
using mediapipe::tool::ProtoUtilLite;
using WireFormatLite = ProtoUtilLite::WireFormatLite;
using FieldValue = ProtoUtilLite::FieldValue;
using FieldType = ProtoUtilLite::FieldType;
using ProtoPath = ProtoUtilLite::ProtoPath;
using ProtoPathEntry = ProtoUtilLite::ProtoPathEntry;

namespace {

// Returns a template argument by name.
TemplateArgument* GetItem(TemplateDict* args, const std::string& name) {
  for (TemplateDict::Parameter& arg : *args->mutable_arg()) {
    if (arg.key() == name) {
      return arg.mutable_value();
    }
  }
  return nullptr;
}

// Sets the template argument for a param name.
void PutItem(TemplateDict* args, const std::string& name,
             const TemplateArgument* value) {
  for (int i = args->arg_size() - 1; i >= 0; --i) {
    if (args->arg()[i].key() == name) {
      if (value != nullptr) {
        *args->mutable_arg(i)->mutable_value() = *value;
      } else {
        args->mutable_arg()->erase(args->mutable_arg()->begin() + i);
      }
      return;
    }
  }
  if (value != nullptr) {
    TemplateDict::Parameter* arg = args->add_arg();
    *arg->mutable_key() = name;
    *arg->mutable_value() = *value;
  }
}

// Creates a deep copy of a message.
std::unique_ptr<MessageLite> CloneMessage(const MessageLite& message) {
  std::unique_ptr<MessageLite> result(message.New());
  result->CheckTypeAndMergeFrom(message);
  return result;
}

// Parses one ProtoPathEntry.
// The parsed entry is appended to `result` and removed from `path`.
// ProtoPathEntry::key_value stores map key text.  Use SetMapKeyTypes
// to serialize the key text to protobuf wire format.
absl::Status ParseEntry(absl::string_view& path, ProtoPath* result) {
  bool ok = true;
  int sb = path.find('[');
  int eb = path.find(']');
  int field_id = -1;
  ok &= absl::SimpleAtoi(path.substr(0, sb), &field_id);
  auto selector = path.substr(sb + 1, eb - 1 - sb);
  if (absl::StartsWith(selector, "@")) {
    int eq = selector.find('=');
    int key_id = -1;
    ok &= absl::SimpleAtoi(selector.substr(1, eq - 1), &key_id);
    auto key_text = selector.substr(eq + 1);
    FieldType key_type = FieldType::TYPE_STRING;
    result->push_back({field_id, key_id, key_type, std::string(key_text)});
  } else {
    int index = 0;
    ok &= absl::SimpleAtoi(selector, &index);
    result->push_back({field_id, index});
  }
  int end = path.find('/', eb);
  if (end == std::string::npos) {
    path = "";
  } else {
    path = path.substr(end + 1);
  }
  return ok ? absl::OkStatus()
            : absl::InvalidArgumentError(
                  absl::StrCat("Failed to parse ProtoPath entry: ", path));
}

// Specifies the FieldTypes for protobuf map keys in a ProtoPath.
// Each ProtoPathEntry::key_value is converted from text to the protobuf
// wire format for its key type.
absl::Status SetMapKeyTypes(const std::vector<FieldType>& key_types,
                            ProtoPath* result) {
  int i = 0;
  for (ProtoPathEntry& entry : *result) {
    if (entry.map_id >= 0) {
      FieldType key_type = key_types[i++];
      std::vector<FieldValue> key_value;
      MP_RETURN_IF_ERROR(
          ProtoUtilLite::Serialize({entry.key_value}, key_type, &key_value));
      entry.key_type = key_type;
      entry.key_value = key_value.front();
    }
  }
  return absl::OkStatus();
}

// Returns the (tag, index) pairs in a field path.
// For example, returns {{1, 1}, {2, 1}, {3, 1}} for "/1[1]/2[1]/3[1]",
// returns {{1, 1}, {2, 1, "INPUT_FRAMES"}} for "/1[1]/2[@1=INPUT_FRAMES]".
absl::Status ProtoPathSplit(const std::string& path, ProtoPath* result) {
  result->clear();
  absl::string_view rest = path;
  if (absl::StartsWith(rest, "/")) {
    rest = rest.substr(1);
  }
  while (!rest.empty()) {
    MP_RETURN_IF_ERROR(ParseEntry(rest, result));
  }
  return absl::OkStatus();
}

// Parse the TemplateExpression.path field into a ProtoPath struct.
absl::Status ParseProtoPath(const TemplateExpression& rule,
                            std::string base_path, ProtoPath* result) {
  ProtoPath base_entries;
  MP_RETURN_IF_ERROR(ProtoPathSplit(base_path, &base_entries));
  MP_RETURN_IF_ERROR(ProtoPathSplit(rule.path(), result));
  std::vector<FieldType> key_types;
  for (int type : rule.key_type()) {
    key_types.push_back(static_cast<FieldType>(type));
  }
  MP_RETURN_IF_ERROR(SetMapKeyTypes(key_types, result));
  result->erase(result->begin(), result->begin() + base_entries.size());
  return absl::OkStatus();
}

// Returns true if one proto path is prefix by another.
bool ProtoPathStartsWith(const std::string& path, const std::string& prefix) {
  return absl::StartsWith(path, prefix);
}

// Returns the target ProtoUtilLite::FieldType of a rule.
FieldType GetFieldType(const TemplateExpression& rule) {
  return static_cast<FieldType>(rule.field_type());
}

// Returns the count of field values at a ProtoPath.
int FieldCount(const FieldValue& base, ProtoPath field_path,
               FieldType field_type) {
  int result = 0;
  ABSL_CHECK_OK(
      ProtoUtilLite::GetFieldCount(base, field_path, field_type, &result));
  return result;
}

}  // namespace

// The default implementation for the mediapipe template rule interpreter.
class TemplateExpanderImpl {
 public:
  explicit TemplateExpanderImpl(std::vector<absl::Status>* errors)
      : errors_(errors) {}

  // Applies the rules specified in a CalculatorGraphTemplate to a
  // CalculatorGraphConfig.  Each rule references a nested field-value or
  // message and defines zero or more replacement values for it.
  bool ExpandTemplates(const TemplateDict& args,
                       const CalculatorGraphTemplate& templ,
                       CalculatorGraphConfig* output) {
    // Extract the serialized CalculatorGraphConfig.
    FieldValue base_value;
    if (!templ.config().SerializeToString(&base_value)) {
      return false;
    }

    // Extract the CalculatorGraphTemplate rules.
    template_rules_ = templ;
    template_rules_.clear_config();

    // Invoke recursive rule expansion.
    environment_ = args;
    std::vector<FieldValue> result;
    if (!ExpandNestedRules(0, "", base_value, &result)) {
      return false;
    }
    return output->ParseFromString(result[0]);
  }

 private:
  // Expands a template rule of a specific type.
  // Modifies a base message to produce one or more expanded messages.
  // Ownership of the result messages is transferred to the caller.
  bool ExpandTemplateRule(int base_index, const FieldValue& base_message,
                          std::vector<FieldValue>* result) {
    // Exapand a template rule of a specific type.
    const TemplateExpression& rule = template_rules_.rule().Get(base_index);
    if (rule.op() == "for") {
      ExpandIterationRule(base_index, base_message, result);
    } else if (rule.op() == "if") {
      ExpandConditionalRule(base_index, base_message, result);
    } else if (rule.op() == "param") {
      ExpandDeclaration(base_index, base_message, result);
    } else {
      ExpandExpressionRule(base_index, result);
    }
    return true;
  }

  // Apply any remaining rules on the current field.
  // If the next rule also applies to the current field, apply it.
  // Otherwise, apply rules for nested fields.
  bool ExpandPeerRules(int base_index, const FieldValue& base_message,
                       std::vector<FieldValue>* result) {
    // If the next rule applies to the same message, apply it now.
    auto& base_rule = template_rules_.rule().Get(base_index);
    int next_index = base_index + 1;
    if (next_index < template_rules_.rule().size()) {
      auto& next_rule = template_rules_.rule().Get(next_index);
      if (next_rule.path() == base_rule.path()) {
        return ExpandTemplateRule(next_index, base_message, result);
      }
    }

    // Otheriwse, apply rules for nested fields.
    return ExpandNestedRules(next_index, base_rule.path(), base_message,
                             result);
  }

  // Return the field values addressed by a template rule.
  absl::Status GetBaseValue(const std::string& base_path,
                            const TemplateExpression& rule,
                            const FieldValue& output,
                            std::vector<FieldValue>* base) {
    if (!rule.has_path()) {
      base->push_back(output);
      return absl::OkStatus();
    }
    if (rule.has_field_value()) {
      // For a non-repeated field, the field value is stored only in the rule.
      base->push_back(rule.field_value());
      return absl::OkStatus();
    }
    ProtoPath field_path;
    MP_RETURN_IF_ERROR(ParseProtoPath(rule, base_path, &field_path));
    return ProtoUtilLite::GetFieldRange(output, field_path, 1,
                                        GetFieldType(rule), base);
  }

  // Replace the field values addressed by a template rule.
  absl::Status ReplaceBaseValue(const std::string& base_path,
                                const TemplateExpression& rule,
                                const std::vector<FieldValue>& field_values,
                                FieldValue* output) {
    if (!rule.has_path()) {
      if (!field_values.empty()) {
        *output = field_values[0];
      }
      return absl::OkStatus();
    }
    ProtoPath field_path;
    MP_RETURN_IF_ERROR(ParseProtoPath(rule, base_path, &field_path));
    int field_count = 1;
    if (rule.has_field_value()) {
      // For a non-repeated field, only one value can be specified.
      if (!field_values.empty() &&
          FieldCount(*output, field_path, GetFieldType(rule)) > 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Multiple values specified for non-repeated field: ", rule.path()));
      }
      // For a non-repeated field, the field value is stored only in the rule.
      field_path[field_path.size() - 1].index = 0;
      field_count = 0;
    }
    return ProtoUtilLite::ReplaceFieldRange(output, field_path, field_count,
                                            GetFieldType(rule), field_values);
  }

  // Replaces nested fields by following nested template rules.
  bool ExpandNestedRules(int base_index, const std::string& base_path,
                         const FieldValue& base_message,
                         std::vector<FieldValue>* result) {
    absl::Status status;
    FieldValue output = base_message;

    // Evaluate the rules nested below base_path in lexical order.
    std::vector<int> rules = GetNestedRules(base_index, base_path);
    std::vector<std::vector<FieldValue>> edits;
    for (int i = 0; i < rules.size(); ++i) {
      const auto& rule = template_rules_.rule().Get(rules[i]);
      std::vector<FieldValue> base;
      status = GetBaseValue(base_path, rule, output, &base);
      if (!status.ok()) break;
      std::vector<FieldValue> values;
      if (!ExpandTemplateRule(rules[i], base[0], &values)) {
        status = absl::InternalError("ExpandTemplateRule failed");
        break;
      }
      edits.push_back(values);
    }
    if (!status.ok()) {
      RecordError(status);
      return false;
    }
    // Replace base field values with the evaluated results.
    // Edits are applied in reverse order since later indices are invalidated.
    for (int i = edits.size() - 1; i >= 0; --i) {
      const auto& rule = template_rules_.rule().Get(rules[i]);
      status = ReplaceBaseValue(base_path, rule, edits[i], &output);
      if (!status.ok()) break;
    }
    if (!status.ok()) {
      RecordError(status);
      return false;
    }
    result->push_back(output);
    return true;
  }

  // Returns indexes of the rules directly nested within a certain rule.
  std::vector<int> GetNestedRules(int rule_index,
                                  const std::string& rule_path) {
    std::vector<int> result;
    std::string prev_path = "-1[-1]";
    for (int i = rule_index; i < template_rules_.rule().size(); ++i) {
      auto& rule = template_rules_.rule().Get(i);
      if (!ProtoPathStartsWith(rule.path(), rule_path)) {
        break;
      }
      if (!ProtoPathStartsWith(rule.path(), prev_path)) {
        result.push_back(i);
        prev_path = rule.path();
      }
    }
    return result;
  }

  // Apply a "for" operation to a base message.
  // Expands nested rules once for each iteration range value.
  bool ExpandIterationRule(int base_index, const FieldValue& base_message,
                           std::vector<FieldValue>* result) {
    // Retrieve the var param and the range expression.
    const TemplateExpression& rule = template_rules_.rule().Get(base_index);
    std::string var_param = rule.arg().Get(0).param();
    const TemplateExpression& range_expr = rule.arg().Get(1);
    TemplateArgument range = EvalExpression(range_expr);

    // For each value of the range param, expand all nested rules.
    TemplateArgument* shadow_item = GetItem(&environment_, var_param);
    for (const TemplateArgument& item : range.element()) {
      PutItem(&environment_, var_param, &item);
      ExpandPeerRules(base_index, base_message, result);
    }
    PutItem(&environment_, var_param, shadow_item);
    return true;
  }

  // Initializes a parameter in the parameter environment.
  bool ExpandDeclaration(int base_index, const FieldValue& base_message,
                         std::vector<FieldValue>* result) {
    // Retrieve the var param and the range expression.
    const TemplateExpression& rule = template_rules_.rule().Get(base_index);
    if (rule.arg().empty() || rule.arg().size() > 2) {
      RecordError(absl::InvalidArgumentError(
          "Param declaration must specify a parameter name and "
          "may specify a single default value."));
    }
    // TODO: Validate that all params are declared or none.
    // Delarations for required params will have no default value.
    if (rule.arg().size() == 2) {
      std::string var_param = rule.arg().Get(0).param();
      const TemplateExpression& item_expr = rule.arg().Get(1);
      TemplateArgument item = EvalExpression(item_expr);
      // The parameter default value is used if no other value is specified.
      if (GetItem(&environment_, var_param) == nullptr) {
        PutItem(&environment_, var_param, &item);
      }
    }
    ExpandPeerRules(base_index, base_message, result);
    return true;
  }

  // Applies an "if" operation to a base message.
  // Expands nested rules zero or more times.
  bool ExpandConditionalRule(int base_index, const FieldValue& base_message,
                             std::vector<FieldValue>* result) {
    // Retrieve the condition expression.
    const TemplateExpression& rule = template_rules_.rule().Get(base_index);
    // Expand this template zero or one times.
    bool condition = AsBool(EvalExpression(rule.arg(0)));
    if (condition) {
      ExpandPeerRules(base_index, base_message, result);
    }
    return true;
  }

  // A self-contained expression just defines a single result value.
  bool ExpandExpressionRule(int base_index, std::vector<FieldValue>* result) {
    const TemplateExpression& rule = template_rules_.rule().Get(base_index);
    TemplateArgument item = EvalExpression(rule);
    std::vector<FieldValue> values;
    absl::Status status = AsFieldValues(std::vector<TemplateArgument>{item},
                                        GetFieldType(rule), &values);
    if (!status.ok()) {
      RecordError(status);
      return false;
    }
    result->push_back(values[0]);
    return true;
  }

  // The "param" operation does variable environment lookup.
  TemplateArgument EvalParam(const TemplateExpression& expr) {
    TemplateArgument* result = GetItem(&environment_, expr.param());
    if (result == nullptr) {
      RecordError(absl::NotFoundError(absl::StrCat("param: ", expr.param())));
      return AsArgument(0.0);
    }
    return *result;
  }

  // The "." operator does template dict lookup.
  TemplateArgument EvalDot(const TemplateExpression& expr) {
    TemplateArgument lhs = EvalExpression(expr.arg(0));
    TemplateArgument* result = GetItem(lhs.mutable_dict(), expr.arg(1).param());
    if (result == nullptr) {
      RecordError(absl::NotFoundError(
          absl::StrCat("param field: ", expr.arg(1).param())));
      return AsArgument(0.0);
    }
    return *result;
  }

  // Converts a TemplateArgument to double.
  double AsNum(const TemplateArgument& value) {
    double result = 0;
    if (value.has_num()) {
      result = value.num();
    }
    if (value.has_str()) {
      if (!absl::SimpleAtod(value.str(), &result)) {
        RecordError(absl::InvalidArgumentError(value.str()));
      }
    }
    return result;
  }

  // Converts a TemplateArgument to string.
  std::string AsString(const TemplateArgument& value) {
    std::string result;
    if (value.has_num()) {
      result = absl::StrCat(value.num());
    }
    if (value.has_str()) {
      result = value.str();
    }
    return result;
  }

  // Converts a TemplateArgument to bool.
  bool AsBool(const TemplateArgument& value) {
    bool result = false;
    if (value.has_num()) {
      return value.num() != 0;
    } else if (value.has_str()) {
      if (!absl::SimpleAtob(value.str(), &result)) {
        RecordError(absl::InvalidArgumentError(value.str()));
      }
    }
    return result;
  }

  // Converts a vector of TemplateArguments to a dict TemplateArgument.
  TemplateArgument AsDict(const std::vector<TemplateArgument>& args) {
    TemplateArgument result;
    if (args.size() % 2 != 0) {
      RecordError(absl::InvalidArgumentError(absl::StrCat(
          "Dict requires an even number of arguments, got: ", args.size())));
      return result;
    }
    TemplateDict* dict = result.mutable_dict();
    for (int i = 0; i < args.size(); i += 2) {
      TemplateDict::Parameter* p = dict->add_arg();
      *p->mutable_key() = AsString(args[i]);
      *p->mutable_value() = args[i + 1];
    }
    return result;
  }

  // Converts a vector of TemplateArguments to a list TemplateArgument.
  TemplateArgument AsList(const std::vector<TemplateArgument>& args) {
    TemplateArgument result;
    auto list = result.mutable_element();
    for (int i = 0; i < args.size(); ++i) {
      *list->Add() = args[i];
    }
    return result;
  }

  // Evaluate each of the sub-expressions of a TemplateExpression.
  void EvalNestedExpressions(const TemplateExpression& expr,
                             std::vector<TemplateArgument>* result) {
    for (const TemplateExpression& e : expr.arg()) {
      result->push_back(EvalExpression(e));
    }
  }

  // Returns true if a TemplateArgument represents a number.
  bool IsNum(const TemplateArgument& value) {
    double r = 0;
    return value.has_num() || absl::SimpleAtod(value.str(), &r);
  }

  // Returns 0 if v1 == v1, positive if v1 > v2, negative if v1 < v2.
  int CompareArgs(const TemplateArgument& v1, const TemplateArgument& v2) {
    if (IsNum(v1) && IsNum(v2)) {
      double d = AsNum(v1) - AsNum(v2);
      return (d < 0) ? -1 : (d > 0) ? 1 : 0;
    } else {
      return AsString(v1).compare(AsString(v2));
    }
  }

  // Evaluates a TemplateExpression to produce a template argument.
  TemplateArgument EvalExpression(const TemplateExpression& expr) {
    if (expr.op() == "literal") {
      return AsArgument(expr.param());
    } else if (expr.op() == ".") {
      return EvalDot(expr);
    } else if (expr.has_param()) {
      return EvalParam(expr);
    }
    std::vector<TemplateArgument> args;
    EvalNestedExpressions(expr, &args);
    TemplateArgument result;
    if (expr.op() == "paren") {
      result = args[0];
    } else if (expr.op() == "+") {
      if (IsNum(args[0]) && IsNum(args[1])) {
        result = AsArgument(AsNum(args[0]) + AsNum(args[1]));
      } else {
        result = AsArgument(AsString(args[0]) + AsString(args[1]));
      }
    } else if (expr.op() == "-") {
      result = AsArgument(AsNum(args[0]) - AsNum(args[1]));
    } else if (expr.op() == "*") {
      result = AsArgument(AsNum(args[0]) * AsNum(args[1]));
    } else if (expr.op() == "/") {
      result = AsArgument(AsNum(args[0]) / AsNum(args[1]));
    } else if (expr.op() == ">") {
      result = AsArgument(CompareArgs(args[0], args[1]) > 0);
    } else if (expr.op() == "<") {
      result = AsArgument(CompareArgs(args[0], args[1]) < 0);
    } else if (expr.op() == ">=") {
      result = AsArgument(CompareArgs(args[0], args[1]) >= 0);
    } else if (expr.op() == "<=") {
      result = AsArgument(CompareArgs(args[0], args[1]) <= 0);
    } else if (expr.op() == "==") {
      result = AsArgument(CompareArgs(args[0], args[1]) == 0);
    } else if (expr.op() == "!=") {
      result = AsArgument(CompareArgs(args[0], args[1]) != 0);
    } else if (expr.op() == "&&") {
      result = AsArgument(AsBool(args[0]) && AsBool(args[1]));
    } else if (expr.op() == "||") {
      result = AsArgument(AsBool(args[0]) || AsBool(args[1]));
    } else if (expr.op() == "!") {
      result = AsArgument(!(AsBool(args[0])));
    } else if (expr.op() == "min") {
      result = AsArgument(std::min(AsNum(args[0]), AsNum(args[1])));
    } else if (expr.op() == "max") {
      result = AsArgument(std::max(AsNum(args[0]), AsNum(args[1])));
    } else if (expr.op() == "concat") {
      result = AsArgument(AsString(args[0]) + AsString(args[1]));
    } else if (expr.op() == "lowercase") {
      result = AsArgument(absl::AsciiStrToLower(AsString(args[0])));
    } else if (expr.op() == "uppercase") {
      result = AsArgument(absl::AsciiStrToUpper(AsString(args[0])));
    } else if (expr.op() == "dict") {
      result = AsDict(args);
    } else if (expr.op() == "list") {
      result = AsList(args);
    } else if (expr.op() == "size") {
      return AsArgument(static_cast<double>(
          args[0].has_dict() ? args[0].mutable_dict()->arg_size()
                             : args[0].mutable_element()->size()));
    }
    return result;
  }

  // Converts a simple value to a template argument for further processing.
  TemplateArgument AsArgument(const std::string& value) {
    TemplateArgument result;
    result.set_str(value);
    return result;
  }

  // Converts a simple value to a template argument for further processing.
  TemplateArgument AsArgument(double value) {
    TemplateArgument result;
    result.set_num(value);
    return result;
  }

  // Converts a boolean result into a template argument for further processing.
  TemplateArgument AsArgument(bool b) {
    return AsArgument(static_cast<double>(b));
  }

  // Convert between a proto field value and a template argument.
  absl::Status AsFieldValues(const std::vector<TemplateArgument>& args,
                             FieldType field_type,
                             std::vector<FieldValue>* result) {
    for (int i = 0; i < args.size(); ++i) {
      if (args[i].has_dict()) {
        FieldValue dict_bytes;
        ABSL_CHECK(args[i].dict().SerializePartialToString(&dict_bytes));
        result->push_back(dict_bytes);
      } else if (args[i].has_num() || args[i].has_str()) {
        std::string text_value = args[i].has_num()
                                     ? mediapipe::SimpleDtoa(args[i].num())
                                     : args[i].str();
        std::vector<FieldValue> r;
        MP_RETURN_IF_ERROR(
            ProtoUtilLite::Serialize({text_value}, field_type, &r));
        result->push_back(r[0]);
      }
    }
    return absl::OkStatus();
  }

  // Record a Status if it indicates an error.
  void RecordError(const absl::Status& status) {
    if (!status.ok()) {
      errors_->push_back(status);
    }
  }

 private:
  // The list of template rules.
  mediapipe::CalculatorGraphTemplate template_rules_;

  // The template variable environment.
  TemplateDict environment_;

  // List of errors found in template parameters.
  std::vector<absl::Status>* errors_;
};

TemplateExpander::TemplateExpander() {}

// Expands template rules within a proto message.
// Replaces template rules with expanded sub-messages.
absl::Status TemplateExpander::ExpandTemplates(
    const TemplateDict& args, const CalculatorGraphTemplate& templ,
    CalculatorGraphConfig* output) {
  errors_.clear();
  TemplateExpanderImpl expander(&errors_);
  if (!expander.ExpandTemplates(args, templ, output)) {
    errors_.push_back(absl::InternalError("ExpandTemplates failed"));
  }
  absl::Status status;
  for (const absl::Status& error : errors_) {
    ABSL_LOG(ERROR) << error;
    status.Update(error);
  }
  return status;
}

}  // namespace tool
}  // namespace mediapipe
