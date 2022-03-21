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

#include "mediapipe/framework/tool/template_parser.h"

#include <limits>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "mediapipe/framework/deps/proto_descriptor.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/calculator_graph_template.pb.h"
#include "mediapipe/framework/tool/proto_util_lite.h"

using mediapipe::proto_ns::Descriptor;
using mediapipe::proto_ns::DynamicMessageFactory;
using mediapipe::proto_ns::EnumDescriptor;
using mediapipe::proto_ns::EnumValueDescriptor;
using mediapipe::proto_ns::FieldDescriptor;
using mediapipe::proto_ns::Message;
using mediapipe::proto_ns::OneofDescriptor;
using mediapipe::proto_ns::Reflection;
using mediapipe::proto_ns::TextFormat;

namespace mediapipe {

namespace tool {

namespace io {

using proto_ns::io::ArrayInputStream;
using proto_ns::io::CodedInputStream;
using proto_ns::io::CodedOutputStream;
using proto_ns::io::ErrorCollector;
using proto_ns::io::StringOutputStream;
using proto_ns::io::Tokenizer;
using proto_ns::io::ZeroCopyInputStream;
typedef proto_ns::io::Tokenizer::Token Token;

float SafeDoubleToFloat(double value) {
  if (value > std::numeric_limits<float>::max()) {
    return std::numeric_limits<float>::infinity();
  } else if (value < -std::numeric_limits<float>::max()) {
    return -std::numeric_limits<float>::infinity();
  } else {
    return static_cast<float>(value);
  }
}

}  // namespace io

namespace internal {
using proto_ns::internal::GetAnyFieldDescriptors;
using proto_ns::internal::kTypeGoogleApisComPrefix;
using proto_ns::internal::kTypeGoogleProdComPrefix;
using proto_ns::internal::WireFormatLite;
}  // namespace internal

namespace {

inline bool IsHexNumber(const std::string& str) {
  return (str.length() >= 2 && str[0] == '0' &&
          (str[1] == 'x' || str[1] == 'X'));
}

inline bool IsOctNumber(const std::string& str) {
  return (str.length() >= 2 && str[0] == '0' &&
          (str[1] >= '0' && str[1] < '8'));
}

// Returns true if two tokens are adjacent with no whitespace separation.
bool IsAdjacent(const io::Token& t1, const io::Token& t2) {
  return t1.line == t2.line && t1.end_column == t2.column;
}

// A tokenizer with support for a few two-symbol tokens.
class Tokenizer {
 public:
  Tokenizer(io::ZeroCopyInputStream* input, io::ErrorCollector* error_collector)
      : tokenizer_(input, error_collector) {
    // For backwards-compatibility with proto1, we need to allow the 'f' suffix
    // for floats.
    tokenizer_.set_allow_f_after_float(true);

    // '#' starts a comment.
    tokenizer_.set_comment_style(io::Tokenizer::SH_COMMENT_STYLE);
    tokenizer_.set_require_space_after_number(false);
    tokenizer_.set_allow_multiline_strings(true);

    // Look ahead one token.
    current_ = tokenizer_.current();
    tokenizer_.Next();
  }

  // Reads the next token, joining two symbols if needed.
  bool Next() {
    static auto kDoubleTokens =
        new std::set<std::string>{">=", "<=", "==", "!=", "&&", "||"};
    current_ = tokenizer_.current();
    tokenizer_.Next();
    if (IsAdjacent(current_, tokenizer_.current())) {
      std::string double_token =
          absl::StrCat(current_.text, tokenizer_.current().text);
      if (kDoubleTokens->count(double_token) > 0) {
        current_.text = double_token;
        current_.end_column = tokenizer_.current().end_column;
        tokenizer_.Next();
      }
    }
    return true;
  }

  // Returns the latest fully resolved token.
  const io::Token& current() { return current_; }

 private:
  // The delegate Tokenizer.
  io::Tokenizer tokenizer_;
  // The latest fully resolved token.
  io::Token current_;
};

}  // namespace

// ===========================================================================
// Implementation of the parse information tree class.
TemplateParser::ParseInfoTree::ParseInfoTree() {}

TemplateParser::ParseInfoTree::~ParseInfoTree() {}

void TemplateParser::ParseInfoTree::RecordLocation(
    const FieldDescriptor* field, TextFormat::ParseLocation location) {
  locations_[field].push_back(location);
}

TemplateParser::ParseInfoTree*
TemplateParser::ParseInfoTree::CreateNested(  // NOLINT
    const FieldDescriptor* field) {
  // Owned by us in the map.
  auto instance = absl::make_unique<TemplateParser::ParseInfoTree>();
  std::vector<std::unique_ptr<TemplateParser::ParseInfoTree>>* trees =
      &nested_[field];
  instance->path_ =
      absl::StrCat(path_, "/", field->number(), "[", trees->size(), "]");
  trees->push_back(std::move(instance));
  return trees->back().get();
}

void CheckFieldIndex(const FieldDescriptor* field, int index) {
  if (field == NULL) {
    return;
  }

  if (field->is_repeated() && index == -1) {
    LOG(DFATAL) << "Index must be in range of repeated field values. "
                << "Field: " << field->name();
  } else if (!field->is_repeated() && index != -1) {
    LOG(DFATAL) << "Index must be -1 for singular fields."
                << "Field: " << field->name();
  }
}

TextFormat::ParseLocation TemplateParser::ParseInfoTree::GetLocation(
    const FieldDescriptor* field, int index) const {
  CheckFieldIndex(field, index);
  if (index == -1) {
    index = 0;
  }

  const std::vector<TextFormat::ParseLocation>* locations =
      mediapipe::FindOrNull(locations_, field);
  if (locations == NULL || index >= locations->size()) {
    return TextFormat::ParseLocation();
  }

  return (*locations)[index];
}

TemplateParser::ParseInfoTree* TemplateParser::ParseInfoTree::GetTreeForNested(
    const FieldDescriptor* field, int index) const {
  CheckFieldIndex(field, index);
  if (index == -1) {
    index = 0;
  }

  const std::vector<std::unique_ptr<TemplateParser::ParseInfoTree>>* trees =
      mediapipe::FindOrNull(nested_, field);
  if (trees == NULL || index >= trees->size()) {
    return NULL;
  }

  return (*trees)[index].get();
}

std::string TemplateParser::ParseInfoTree::GetLastPath(
    const FieldDescriptor* field) {
  int index = locations_[field].size();
  return absl::StrCat(path_, "/", field->number(), "[", index, "]");
}

std::string TemplateParser::ParseInfoTree::GetPath() { return path_; }

namespace {
// These functions implement the behavior of the "default" TextFormat::Finder,
// they are defined as standalone to be called when finder_ is NULL.
const FieldDescriptor* DefaultFinderFindExtension(Message* message,
                                                  const std::string& name) {
  return message->GetReflection()->FindKnownExtensionByName(name);
}

const Descriptor* DefaultFinderFindAnyType(const Message& message,
                                           const std::string& prefix,
                                           const std::string& name) {
  if (prefix != internal::kTypeGoogleApisComPrefix &&
      prefix != internal::kTypeGoogleProdComPrefix) {
    return NULL;
  }
  return message.GetDescriptor()->file()->pool()->FindMessageTypeByName(name);
}
}  // namespace

// ===========================================================================
// Internal class for parsing an ASCII representation of a Protocol Message.

// Makes code slightly more readable.  The meaning of "DO(foo)" is
// "Execute foo and fail if it fails.", where failure is indicated by
// returning false. Borrowed from parser.cc (Thanks Kenton!).
#define DO(STATEMENT) \
  if (STATEMENT) {    \
  } else {            \
    return false;     \
  }

class TemplateParser::Parser::ParserImpl {
 public:
  typedef proto_ns::TextFormat::ParseLocation ParseLocation;

  // Determines if repeated values for non-repeated fields and
  // oneofs are permitted, e.g., the string "foo: 1 foo: 2" for a
  // required/optional field named "foo", or "baz: 1 qux: 2"
  // where "baz" and "qux" are members of the same oneof.
  enum SingularOverwritePolicy {
    ALLOW_SINGULAR_OVERWRITES = 0,   // the last value is retained
    FORBID_SINGULAR_OVERWRITES = 1,  // an error is issued
  };

  ParserImpl(const Descriptor* root_message_type,
             io::ZeroCopyInputStream* input_stream,
             io::ErrorCollector* error_collector,
             const TextFormat::Finder* finder, ParseInfoTree* parse_info_tree,
             SingularOverwritePolicy singular_overwrite_policy,
             bool allow_case_insensitive_field, bool allow_unknown_field,
             bool allow_unknown_extension, bool allow_unknown_enum,
             bool allow_field_number, bool allow_relaxed_whitespace,
             bool allow_partial, int recursion_limit)
      : error_collector_(error_collector),
        finder_(finder),
        parse_info_tree_(parse_info_tree),
        tokenizer_error_collector_(this),
        tokenizer_(input_stream, &tokenizer_error_collector_),
        root_message_type_(root_message_type),
        singular_overwrite_policy_(singular_overwrite_policy),
        allow_case_insensitive_field_(allow_case_insensitive_field),
        allow_unknown_field_(allow_unknown_field),
        allow_unknown_extension_(allow_unknown_extension),
        allow_unknown_enum_(allow_unknown_enum),
        allow_field_number_(allow_field_number),
        allow_partial_(allow_partial),
        recursion_limit_(recursion_limit),
        had_errors_(false) {
    // Consume the starting token.
    tokenizer_.Next();
  }
  virtual ~ParserImpl() {}

  // Parses the ASCII representation specified in input and saves the
  // information into the output pointer (a Message). Returns
  // false if an error occurs (an error will also be logged to
  // LOG(ERROR)).
  virtual bool Parse(Message* output) {
    // Consume fields until we cannot do so anymore.
    while (true) {
      if (LookingAtType(io::Tokenizer::TYPE_END)) {
        return !had_errors_;
      }

      if (LookingAt("%")) {
        DO(ConsumeFieldTemplate(output));
      } else {
        DO(ConsumeField(output));
      }
    }
  }

  bool ParseField(const FieldDescriptor* field, Message* output) {
    bool suc;
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      suc = ConsumeFieldMessage(output, output->GetReflection(), field);
    } else {
      suc = ConsumeFieldValue(output, output->GetReflection(), field);
    }
    return suc && LookingAtType(io::Tokenizer::TYPE_END);
  }

  void ReportError(int line, int col, const std::string& message) {
    had_errors_ = true;
    if (error_collector_ == NULL) {
      if (line >= 0) {
        LOG(ERROR) << "Error parsing text-format "
                   << root_message_type_->full_name() << ": " << (line + 1)
                   << ":" << (col + 1) << ": " << message;
      } else {
        LOG(ERROR) << "Error parsing text-format "
                   << root_message_type_->full_name() << ": " << message;
      }
    } else {
      error_collector_->AddError(line, col, message);
    }
  }

  void ReportWarning(int line, int col, const std::string& message) {
    if (error_collector_ == NULL) {
      if (line >= 0) {
        LOG(WARNING) << "Warning parsing text-format "
                     << root_message_type_->full_name() << ": " << (line + 1)
                     << ":" << (col + 1) << ": " << message;
      } else {
        LOG(WARNING) << "Warning parsing text-format "
                     << root_message_type_->full_name() << ": " << message;
      }
    } else {
      error_collector_->AddWarning(line, col, message);
    }
  }

 protected:
  // Reports an error with the given message with information indicating
  // the position (as derived from the current token).
  void ReportError(const std::string& message) {
    ReportError(tokenizer_.current().line, tokenizer_.current().column,
                message);
  }

  // Reports a warning with the given message with information indicating
  // the position (as derived from the current token).
  void ReportWarning(const std::string& message) {
    ReportWarning(tokenizer_.current().line, tokenizer_.current().column,
                  message);
  }

  // Consumes the specified message with the given starting delimiter.
  // This method checks to see that the end delimiter at the conclusion of
  // the consumption matches the starting delimiter passed in here.
  bool ConsumeMessage(Message* message, const std::string delimiter) {
    while (!LookingAt(">") && !LookingAt("}")) {
      if (LookingAt("%")) {
        DO(ConsumeFieldTemplate(message));
      } else {
        DO(ConsumeField(message));
      }
    }

    // Confirm that we have a valid ending delimiter.
    DO(Consume(delimiter));
    return true;
  }

  // Consume either "<" or "{".
  bool ConsumeMessageDelimiter(std::string* delimiter) {
    if (TryConsume("<")) {
      *delimiter = ">";
    } else {
      DO(Consume("{"));
      *delimiter = "}";
    }
    return true;
  }

#ifndef PROTO2_OPENSOURCE
  // Consumes a string value and parses it as a packed repeated field into
  // the given field of the given message.
  bool ConsumePackedFieldAsString(const std::string& field_name,
                                  const FieldDescriptor* field,
                                  Message* message) {
    std::string packed;
    DO(ConsumeString(&packed));

    // Prepend field tag and varint-encoded string length to turn into
    // encoded message.
    std::string tagged;
    {
      io::StringOutputStream string_output(&tagged);
      io::CodedOutputStream coded_output(&string_output);
      // Force length-delimited format, even if field not currently packed.
      coded_output.WriteTag(internal::WireFormatLite::MakeTag(
          field->number(),
          internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED));
      coded_output.WriteVarint32(packed.size());
      coded_output.WriteString(packed);
    }

    // Merge encoded message.
    io::ArrayInputStream array_input(tagged.data(), tagged.size());
    io::CodedInputStream coded_input(&array_input);
    if (!message->MergePartialFromCodedStream(&coded_input)) {
      ReportError("Could not parse packed field \"" + field_name +
                  "\" as wire-encoded string.");
      return false;
    }

    return true;
  }
#endif  // !PROTO2_OPENSOURCE

  // Consumes the current field (as returned by the tokenizer) on the
  // passed in message.
  bool ConsumeField(Message* message) {
    const Reflection* reflection = message->GetReflection();
    const Descriptor* descriptor = message->GetDescriptor();

    std::string field_name;
    bool reserved_field = false;
    const FieldDescriptor* field = NULL;
    int start_line = tokenizer_.current().line;
    int start_column = tokenizer_.current().column;

    const FieldDescriptor* any_type_url_field;
    const FieldDescriptor* any_value_field;
    if (internal::GetAnyFieldDescriptors(*message, &any_type_url_field,
                                         &any_value_field) &&
        TryConsume("[")) {
      std::string full_type_name, prefix;
      DO(ConsumeAnyTypeUrl(&full_type_name, &prefix));
      DO(Consume("]"));
      TryConsume(":");  // ':' is optional between message labels and values.
      std::string serialized_value;
      const Descriptor* value_descriptor =
          finder_ ? finder_->FindAnyType(*message, prefix, full_type_name)
                  : DefaultFinderFindAnyType(*message, prefix, full_type_name);
      if (value_descriptor == NULL) {
        ReportError("Could not find type \"" + prefix + full_type_name +
                    "\" stored in google.protobuf.Any.");
        return false;
      }
      DO(ConsumeAnyValue(value_descriptor, &serialized_value));
      if (singular_overwrite_policy_ == FORBID_SINGULAR_OVERWRITES) {
        // Fail if any_type_url_field has already been specified.
        if ((!any_type_url_field->is_repeated() &&
             reflection->HasField(*message, any_type_url_field)) ||
            (!any_value_field->is_repeated() &&
             reflection->HasField(*message, any_value_field))) {
          ReportError("Non-repeated Any specified multiple times.");
          return false;
        }
      }
      reflection->SetString(message, any_type_url_field,
                            std::string(prefix + full_type_name));
      reflection->SetString(message, any_value_field, serialized_value);
      return true;
    }
    if (TryConsume("[")) {
      // Extension.
      DO(ConsumeFullTypeName(&field_name));
      DO(Consume("]"));

      field = finder_ ? finder_->FindExtension(message, field_name)
                      : DefaultFinderFindExtension(message, field_name);

      if (field == NULL) {
        if (!allow_unknown_field_ && !allow_unknown_extension_) {
          ReportError("Extension \"" + field_name +
                      "\" is not defined or "
                      "is not an extension of \"" +
                      descriptor->full_name() + "\".");
          return false;
        } else {
          ReportWarning("Ignoring extension \"" + field_name +
                        "\" which is not defined or is not an extension of \"" +
                        descriptor->full_name() + "\".");
        }
      }
    } else {
      DO(ConsumeIdentifier(&field_name));

      if (allow_field_number_) {
        int32 field_number = std::atoi(field_name.c_str());  // NOLINT
        if (descriptor->IsExtensionNumber(field_number)) {
          field = reflection->FindKnownExtensionByNumber(field_number);
        } else if (descriptor->IsReservedNumber(field_number)) {
          reserved_field = true;
        } else {
          field = descriptor->FindFieldByNumber(field_number);
        }
      } else {
        field = descriptor->FindFieldByName(field_name);
        // Group names are expected to be capitalized as they appear in the
        // .proto file, which actually matches their type names, not their
        // field names.
        if (field == NULL) {
          std::string lower_field_name = field_name;
          absl::AsciiStrToLower(&lower_field_name);
          field = descriptor->FindFieldByName(lower_field_name);
          // If the case-insensitive match worked but the field is NOT a group,
          if (field != NULL && field->type() != FieldDescriptor::TYPE_GROUP) {
            field = NULL;
          }
        }
        // Again, special-case group names as described above.
        if (field != NULL && field->type() == FieldDescriptor::TYPE_GROUP &&
            field->message_type()->name() != field_name) {
          field = NULL;
        }

        if (field == NULL && allow_case_insensitive_field_) {
          std::string lower_field_name = field_name;
          absl::AsciiStrToLower(&lower_field_name);
          field = descriptor->FindFieldByLowercaseName(lower_field_name);
        }

        if (field == NULL) {
          reserved_field = descriptor->IsReservedName(field_name);
        }
      }

      if (field == NULL && !reserved_field) {
        if (!allow_unknown_field_) {
          ReportError("Message type \"" + descriptor->full_name() +
                      "\" has no field named \"" + field_name + "\".");
          return false;
        } else {
          ReportWarning("Message type \"" + descriptor->full_name() +
                        "\" has no field named \"" + field_name + "\".");
        }
      }
    }

    // Skips unknown or reserved fields.
    if (field == NULL) {
      CHECK(allow_unknown_field_ || allow_unknown_extension_ || reserved_field);

      // Try to guess the type of this field.
      // If this field is not a message, there should be a ":" between the
      // field name and the field value and also the field value should not
      // start with "{" or "<" which indicates the beginning of a message body.
      // If there is no ":" or there is a "{" or "<" after ":", this field has
      // to be a message or the input is ill-formed.
      if (TryConsume(":") && !LookingAt("{") && !LookingAt("<")) {
        return SkipFieldValue();
      } else {
        return SkipFieldMessage();
      }
    }

    if (singular_overwrite_policy_ == FORBID_SINGULAR_OVERWRITES) {
      // Fail if the field is not repeated and it has already been specified.
      if (!field->is_repeated() && reflection->HasField(*message, field)) {
        ReportError("Non-repeated field \"" + field_name +
                    "\" is specified multiple times.");
        return false;
      }
      // Fail if the field is a member of a oneof and another member has already
      // been specified.
      const OneofDescriptor* oneof = field->containing_oneof();
      if (oneof != NULL && reflection->HasOneof(*message, oneof)) {
        const FieldDescriptor* other_field =
            reflection->GetOneofFieldDescriptor(*message, oneof);
        ReportError("Field \"" + field_name +
                    "\" is specified along with "
                    "field \"" +
                    other_field->name() +
                    "\", another member "
                    "of oneof \"" +
                    oneof->name() + "\".");
        return false;
      }
    }

    // MediaPipe: Update the field path.
    EnterField(field);
    // Perform special handling for embedded message types.
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      // ':' is optional here.
      bool consumed_semicolon = TryConsume(":");
      if (consumed_semicolon && field->options().weak() &&
          LookingAtType(io::Tokenizer::TYPE_STRING)) {
        // we are getting a bytes string for a weak field.
        std::string tmp;
        DO(ConsumeString(&tmp));
        reflection->MutableMessage(message, field)->ParseFromString(tmp);
        goto label_skip_parsing;
      }
    } else {
      // ':' is required here.
      DO(Consume(":"));
    }

    if (field->is_repeated() && TryConsume("[")) {
      // Short repeated format, e.g.  "foo: [1, 2, 3]".
      if (!TryConsume("]")) {
        // "foo: []" is treated as empty.
        while (true) {
          if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
            // Perform special handling for embedded message types.
            DO(ConsumeFieldMessage(message, reflection, field));
          } else {
            DO(ConsumeFieldValue(message, reflection, field));
          }
          if (TryConsume("]")) {
            break;
          }
          DO(Consume(","));
        }
      }
    } else if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      DO(ConsumeFieldMessage(message, reflection, field));
#ifndef PROTO2_OPENSOURCE
    } else if (field->is_packable() &&
               LookingAtType(io::Tokenizer::TYPE_STRING)) {
      // Packable field printed as wire-formatted string: "foo: "abc\123"".
      // Fields of type string cannot be packed themselves, so this is
      // unambiguous.
      DO(ConsumePackedFieldAsString(field_name, field, message));
#endif  // !PROTO2_OPENSOURCE
    } else {
      DO(ConsumeFieldValue(message, reflection, field));
    }
  label_skip_parsing:
    // For historical reasons, fields may optionally be separated by commas or
    // semicolons.
    TryConsume(";") || TryConsume(",");

    if (field->options().deprecated()) {
      ReportWarning("text format contains deprecated field \"" + field_name +
                    "\"");
    }

    // If a parse info tree exists, add the location for the parsed
    // field.
    if (parse_info_tree_ != NULL) {
      parse_info_tree_->RecordLocation(field,
                                       ParseLocation(start_line, start_column));
    }
    return true;
  }

  // Skips the next field including the field's name and value.
  bool SkipField() {
    if (TryConsume("[")) {
      // Extension name or type URL.
      DO(ConsumeTypeUrlOrFullTypeName());
      DO(Consume("]"));
    } else {
      std::string field_name;
      DO(ConsumeIdentifier(&field_name));
    }

    // Try to guess the type of this field.
    // If this field is not a message, there should be a ":" between the
    // field name and the field value and also the field value should not
    // start with "{" or "<" which indicates the beginning of a message body.
    // If there is no ":" or there is a "{" or "<" after ":", this field has
    // to be a message or the input is ill-formed.
    if (TryConsume(":") && !LookingAt("{") && !LookingAt("<")) {
      DO(SkipFieldValue());
    } else {
      DO(SkipFieldMessage());
    }
    // For historical reasons, fields may optionally be separated by commas or
    // semicolons.
    TryConsume(";") || TryConsume(",");
    return true;
  }

  bool ConsumeFieldMessage(Message* message, const Reflection* reflection,
                           const FieldDescriptor* field) {
    if (--recursion_limit_ < 0) {
      ReportError("Message is too deep");
      return false;
    }
    // If the parse information tree is not NULL, create a nested one
    // for the nested message.
    ParseInfoTree* parent = parse_info_tree_;
    if (parent != NULL) {
      parse_info_tree_ = parent->CreateNested(field);
    }

    // MediaPipe: message value template.
    if (LookingAt("%")) {
      DO(ConsumeMessageTemplate(message, reflection, field));
    } else {
      std::string delimiter;
      DO(ConsumeMessageDelimiter(&delimiter));
      if (field->is_repeated()) {
        DO(ConsumeMessage(reflection->AddMessage(message, field), delimiter));
      } else {
        DO(ConsumeMessage(reflection->MutableMessage(message, field),
                          delimiter));
      }
    }

    ++recursion_limit_;

    // Reset the parse information tree.
    parse_info_tree_ = parent;
    return true;
  }

  // Skips the whole body of a message including the beginning delimiter and
  // the ending delimiter.
  bool SkipFieldMessage() {
    std::string delimiter;
    DO(ConsumeMessageDelimiter(&delimiter));
    while (!LookingAt(">") && !LookingAt("}")) {
      DO(SkipField());
    }
    DO(Consume(delimiter));
    return true;
  }

  bool ConsumeFieldValue(Message* message, const Reflection* reflection,
                         const FieldDescriptor* field) {
    // MediaPipe simple value template.
    if (LookingAt("%")) {
      DO(ConsumeValueTemplate(message, reflection, field));
      return true;
    }

// Define an easy to use macro for setting fields. This macro checks
// to see if the field is repeated (in which case we need to use the Add
// methods or not (in which case we need to use the Set methods).
#define SET_FIELD(CPPTYPE, VALUE)                    \
  if (field->is_repeated()) {                        \
    reflection->Add##CPPTYPE(message, field, VALUE); \
  } else {                                           \
    reflection->Set##CPPTYPE(message, field, VALUE); \
  }

    switch (field->cpp_type()) {
      case FieldDescriptor::CPPTYPE_INT32: {
        int64 value;
        DO(ConsumeSignedInteger(&value, kint32max));
        SET_FIELD(Int32, static_cast<int32>(value));
        break;
      }

      case FieldDescriptor::CPPTYPE_UINT32: {
        uint64 value;
        DO(ConsumeUnsignedInteger(&value, kuint32max));
        SET_FIELD(UInt32, static_cast<uint32>(value));
        break;
      }

      case FieldDescriptor::CPPTYPE_INT64: {
        int64 value;
        DO(ConsumeSignedInteger(&value, kint64max));
        SET_FIELD(Int64, value);
        break;
      }

      case FieldDescriptor::CPPTYPE_UINT64: {
        uint64 value;
        DO(ConsumeUnsignedInteger(&value, kuint64max));
        SET_FIELD(UInt64, value);
        break;
      }

      case FieldDescriptor::CPPTYPE_FLOAT: {
        double value;
        DO(ConsumeDouble(&value));
        SET_FIELD(Float, io::SafeDoubleToFloat(value));
        break;
      }

      case FieldDescriptor::CPPTYPE_DOUBLE: {
        double value;
        DO(ConsumeDouble(&value));
        SET_FIELD(Double, value);
        break;
      }

      case FieldDescriptor::CPPTYPE_STRING: {
        std::string value;
        DO(ConsumeString(&value));
        SET_FIELD(String, value);
        break;
      }

      case FieldDescriptor::CPPTYPE_BOOL: {
        if (LookingAtType(io::Tokenizer::TYPE_INTEGER)) {
          uint64 value;
          DO(ConsumeUnsignedInteger(&value, 1));
          SET_FIELD(Bool, value);
        } else {
          std::string value;
          DO(ConsumeIdentifier(&value));
          if (value == "true" || value == "True" || value == "t") {
            SET_FIELD(Bool, true);
          } else if (value == "false" || value == "False" || value == "f") {
            SET_FIELD(Bool, false);
          } else {
            ReportError("Invalid value for boolean field \"" + field->name() +
                        "\". Value: \"" + value + "\".");
            return false;
          }
        }
        break;
      }

      case FieldDescriptor::CPPTYPE_ENUM: {
        std::string value;
        int64 int_value = kint64max;
        const EnumDescriptor* enum_type = field->enum_type();
        const EnumValueDescriptor* enum_value = NULL;

        if (LookingAtType(io::Tokenizer::TYPE_IDENTIFIER)) {
          DO(ConsumeIdentifier(&value));
          // Find the enumeration value.
          enum_value = enum_type->FindValueByName(value);

        } else if (LookingAt("-") ||
                   LookingAtType(io::Tokenizer::TYPE_INTEGER)) {
          DO(ConsumeSignedInteger(&int_value, kint32max));
          value = absl::StrCat(int_value);  // for error reporting
          enum_value = enum_type->FindValueByNumber(int_value);
        } else {
          ReportError("Expected integer or identifier, got: " +
                      tokenizer_.current().text);
          return false;
        }

        if (enum_value == NULL) {
          if (int_value != kint64max &&
              reflection->SupportsUnknownEnumValues()) {
            SET_FIELD(EnumValue, int_value);
            return true;
          } else if (!allow_unknown_enum_) {
            ReportError("Unknown enumeration value of \"" + value +
                        "\" for "
                        "field \"" +
                        field->name() + "\".");
            return false;
          } else {
            ReportWarning("Unknown enumeration value of \"" + value +
                          "\" for "
                          "field \"" +
                          field->name() + "\".");
            return true;
          }
        }

        SET_FIELD(Enum, enum_value);
        break;
      }

      case FieldDescriptor::CPPTYPE_MESSAGE: {
        // We should never get here. Put here instead of a default
        // so that if new types are added, we get a nice compiler warning.
        LOG(FATAL) << "Reached an unintended state: CPPTYPE_MESSAGE";
        break;
      }
    }
#undef SET_FIELD
    return true;
  }

  bool SkipFieldValue() {
    if (LookingAtType(io::Tokenizer::TYPE_STRING)) {
      while (LookingAtType(io::Tokenizer::TYPE_STRING)) {
        tokenizer_.Next();
      }
      return true;
    }
    if (TryConsume("[")) {
      while (true) {
        if (!LookingAt("{") && !LookingAt("<")) {
          DO(SkipFieldValue());
        } else {
          DO(SkipFieldMessage());
        }
        if (TryConsume("]")) {
          break;
        }
        DO(Consume(","));
      }
      return true;
    }
    // Possible field values other than string:
    //   12345        => TYPE_INTEGER
    //   -12345       => TYPE_SYMBOL + TYPE_INTEGER
    //   1.2345       => TYPE_FLOAT
    //   -1.2345      => TYPE_SYMBOL + TYPE_FLOAT
    //   inf          => TYPE_IDENTIFIER
    //   -inf         => TYPE_SYMBOL + TYPE_IDENTIFIER
    //   TYPE_INTEGER => TYPE_IDENTIFIER
    // Divides them into two group, one with TYPE_SYMBOL
    // and the other without:
    //   Group one:
    //     12345        => TYPE_INTEGER
    //     1.2345       => TYPE_FLOAT
    //     inf          => TYPE_IDENTIFIER
    //     TYPE_INTEGER => TYPE_IDENTIFIER
    //   Group two:
    //     -12345       => TYPE_SYMBOL + TYPE_INTEGER
    //     -1.2345      => TYPE_SYMBOL + TYPE_FLOAT
    //     -inf         => TYPE_SYMBOL + TYPE_IDENTIFIER
    // As we can see, the field value consists of an optional '-' and one of
    // TYPE_INTEGER, TYPE_FLOAT and TYPE_IDENTIFIER.
    bool has_minus = TryConsume("-");
    if (!LookingAtType(io::Tokenizer::TYPE_INTEGER) &&
        !LookingAtType(io::Tokenizer::TYPE_FLOAT) &&
        !LookingAtType(io::Tokenizer::TYPE_IDENTIFIER)) {
      std::string text = tokenizer_.current().text;
      ReportError("Cannot skip field value, unexpected token: " + text);
      return false;
    }
    // Combination of '-' and TYPE_IDENTIFIER may result in an invalid field
    // value while other combinations all generate valid values.
    // We check if the value of this combination is valid here.
    // TYPE_IDENTIFIER after a '-' should be one of the float values listed
    // below:
    //   inf, inff, infinity, nan
    if (has_minus && LookingAtType(io::Tokenizer::TYPE_IDENTIFIER)) {
      std::string text = tokenizer_.current().text;
      absl::AsciiStrToLower(&text);
      if (text != "inf" &&
#ifndef PROTO2_OPENSOURCE
          text != "inff" &&
#endif  // !PROTO2_OPENSOURCE
          text != "infinity" && text != "nan") {
        ReportError("Invalid float number: " + text);
        return false;
      }
    }
    tokenizer_.Next();
    return true;
  }

  // Returns true if the current token's text is equal to that specified.
  bool LookingAt(const std::string& text) {
    return tokenizer_.current().text == text;
  }

  // Returns true if the current token's type is equal to that specified.
  bool LookingAtType(io::Tokenizer::TokenType token_type) {
    return tokenizer_.current().type == token_type;
  }

  // Consumes an identifier and saves its value in the identifier parameter.
  // Returns false if the token is not of type IDENTFIER.
  bool ConsumeIdentifier(std::string* identifier) {
    if (LookingAtType(io::Tokenizer::TYPE_IDENTIFIER)) {
      *identifier = tokenizer_.current().text;
      tokenizer_.Next();
      return true;
    }

    // If allow_field_numer_ or allow_unknown_field_ is true, we should able
    // to parse integer identifiers.
    if ((allow_field_number_ || allow_unknown_field_ ||
         allow_unknown_extension_) &&
        LookingAtType(io::Tokenizer::TYPE_INTEGER)) {
      *identifier = tokenizer_.current().text;
      tokenizer_.Next();
      return true;
    }

    ReportError("Expected identifier, got: " + tokenizer_.current().text);
    return false;
  }

  // Consume a string of form "<id1>.<id2>....<idN>".
  bool ConsumeFullTypeName(std::string* name) {
    DO(ConsumeIdentifier(name));
    while (TryConsume(".")) {
      std::string part;
      DO(ConsumeIdentifier(&part));
      *name += ".";
      *name += part;
    }
    return true;
  }

  bool ConsumeTypeUrlOrFullTypeName() {
    std::string discarded;
    DO(ConsumeIdentifier(&discarded));
    while (TryConsume(".") || TryConsume("/")) {
      DO(ConsumeIdentifier(&discarded));
    }
    return true;
  }

  // Consumes a string and saves its value in the text parameter.
  // Returns false if the token is not of type STRING.
  bool ConsumeString(std::string* text) {
    if (!LookingAtType(io::Tokenizer::TYPE_STRING)) {
      ReportError("Expected string, got: " + tokenizer_.current().text);
      return false;
    }

    text->clear();
    while (LookingAtType(io::Tokenizer::TYPE_STRING)) {
      io::Tokenizer::ParseStringAppend(tokenizer_.current().text, text);

      tokenizer_.Next();
    }

    return true;
  }

  // Consumes a uint64 and saves its value in the value parameter.
  // Returns false if the token is not of type INTEGER.
  bool ConsumeUnsignedInteger(uint64* value, uint64 max_value) {
    if (!LookingAtType(io::Tokenizer::TYPE_INTEGER)) {
      ReportError("Expected integer, got: " + tokenizer_.current().text);
      return false;
    }

    if (!io::Tokenizer::ParseInteger(tokenizer_.current().text, max_value,
                                     value)) {
      ReportError("Integer out of range (" + tokenizer_.current().text + ")");
      return false;
    }

    tokenizer_.Next();
    return true;
  }

  // Consumes an int64 and saves its value in the value parameter.
  // Note that since the tokenizer does not support negative numbers,
  // we actually may consume an additional token (for the minus sign) in this
  // method. Returns false if the token is not an integer
  // (signed or otherwise).
  bool ConsumeSignedInteger(int64* value, uint64 max_value) {
    bool negative = false;
#ifndef PROTO2_OPENSOURCE
    if (absl::StartsWith(tokenizer_.current().text, "0x")) {
      // proto1 text format allows negative numbers be printed as large positive
      // hex values. We accept these values for backward compatibility.
      max_value = (max_value << 1) + 1;
    }
#endif  // !PROTO2_OPENSOURCE

    if (TryConsume("-")) {
      negative = true;
      // Two's complement always allows one more negative integer than
      // positive.
      ++max_value;
    }

    uint64 unsigned_value;

    DO(ConsumeUnsignedInteger(&unsigned_value, max_value));

    if (negative) {
      if ((static_cast<uint64>(kint64max) + 1) == unsigned_value) {
        *value = kint64min;
      } else {
        *value = -static_cast<int64>(unsigned_value);
      }
    } else {
      *value = static_cast<int64>(unsigned_value);
    }

    return true;
  }

  // Consumes a uint64 and saves its value in the value parameter.
  // Accepts decimal numbers only, rejects hex or oct numbers.
  bool ConsumeUnsignedDecimalInteger(uint64* value, uint64 max_value) {
    if (!LookingAtType(io::Tokenizer::TYPE_INTEGER)) {
      ReportError("Expected integer, got: " + tokenizer_.current().text);
      return false;
    }

    const std::string& text = tokenizer_.current().text;
    if (IsHexNumber(text) || IsOctNumber(text)) {
      ReportError("Expect a decimal number, got: " + text);
      return false;
    }

    if (!io::Tokenizer::ParseInteger(text, max_value, value)) {
      ReportError("Integer out of range (" + text + ")");
      return false;
    }

    tokenizer_.Next();
    return true;
  }

  // Consumes a double and saves its value in the value parameter.
  // Note that since the tokenizer does not support negative numbers,
  // we actually may consume an additional token (for the minus sign) in this
  // method. Returns false if the token is not a double
  // (signed or otherwise).
  bool ConsumeDouble(double* value) {
    bool negative = false;

    if (TryConsume("-")) {
      negative = true;
    }

    // A double can actually be an integer, according to the tokenizer.
    // Therefore, we must check both cases here.
    if (LookingAtType(io::Tokenizer::TYPE_INTEGER)) {
      // We have found an integer value for the double.
      uint64 integer_value;
      DO(ConsumeUnsignedDecimalInteger(&integer_value, kuint64max));

      *value = static_cast<double>(integer_value);
    } else if (LookingAtType(io::Tokenizer::TYPE_FLOAT)) {
      // We have found a float value for the double.
      *value = io::Tokenizer::ParseFloat(tokenizer_.current().text);

      // Mark the current token as consumed.
      tokenizer_.Next();
    } else if (LookingAtType(io::Tokenizer::TYPE_IDENTIFIER)) {
      std::string text = tokenizer_.current().text;
      absl::AsciiStrToLower(&text);
      if (text == "inf" ||
#ifndef PROTO2_OPENSOURCE
          text == "inff" ||
#endif  // !PROTO2_OPENSOURCE
          text == "infinity") {
        *value = std::numeric_limits<double>::infinity();
        tokenizer_.Next();
      } else if (text == "nan") {
        *value = std::numeric_limits<double>::quiet_NaN();
        tokenizer_.Next();
      } else {
        ReportError("Expected double, got: " + text);
        return false;
      }
    } else {
      ReportError("Expected double, got: " + tokenizer_.current().text);
      return false;
    }

    if (negative) {
      *value = -*value;
    }

    return true;
  }

  // Consumes Any::type_url value, of form "type.googleapis.com/full.type.Name"
  // or "type.googleprod.com/full.type.Name"
  bool ConsumeAnyTypeUrl(std::string* full_type_name, std::string* prefix) {
    // TODO Extend Consume() to consume multiple tokens at once, so that
    // this code can be written as just DO(Consume(kGoogleApisTypePrefix)).
    DO(ConsumeIdentifier(prefix));
    while (TryConsume(".")) {
      std::string url;
      DO(ConsumeIdentifier(&url));
      *prefix += "." + url;
    }
    DO(Consume("/"));
    *prefix += "/";
    DO(ConsumeFullTypeName(full_type_name));

    return true;
  }

  // A helper function for reconstructing Any::value. Consumes a text of
  // full_type_name, then serializes it into serialized_value.
  bool ConsumeAnyValue(const Descriptor* value_descriptor,
                       std::string* serialized_value) {
    DynamicMessageFactory factory;
    const Message* value_prototype = factory.GetPrototype(value_descriptor);
    if (value_prototype == NULL) {
      return false;
    }
    std::unique_ptr<Message> value(value_prototype->New());
    std::string sub_delimiter;
    DO(ConsumeMessageDelimiter(&sub_delimiter));
    DO(ConsumeMessage(value.get(), sub_delimiter));

    if (allow_partial_) {
      value->AppendPartialToString(serialized_value);
    } else {
      if (!value->IsInitialized()) {
        ReportError(
            "Value of type \"" + value_descriptor->full_name() +
            "\" stored in google.protobuf.Any has missing required fields");
        return false;
      }
      value->AppendToString(serialized_value);
    }
    return true;
  }

  // Consumes a token and confirms that it matches that specified in the
  // value parameter. Returns false if the token found does not match that
  // which was specified.
  bool Consume(const std::string& value) {
    const std::string& current_value = tokenizer_.current().text;

    if (current_value != value) {
      ReportError("Expected \"" + value + "\", found \"" + current_value +
                  "\".");
      return false;
    }

    tokenizer_.Next();

    return true;
  }

  // Attempts to consume the supplied value. Returns false if a the
  // token found does not match the value specified.
  bool TryConsume(const std::string& value) {
    if (tokenizer_.current().text == value) {
      tokenizer_.Next();
      return true;
    } else {
      return false;
    }
  }

  // Called when parsing for a field begins.
  virtual void EnterField(const FieldDescriptor* field) {}

  // Parse and record a template definition for the current field path.
  virtual bool ConsumeFieldTemplate(Message* message) { return true; }

  // Parse and record a template definition for the current field path.
  virtual bool ConsumeMessageTemplate(Message* message,
                                      const Reflection* reflection,
                                      const FieldDescriptor* field) {
    return true;
  }

  // Parse and record a template definition for the current field path.
  virtual bool ConsumeValueTemplate(Message* message,
                                    const Reflection* reflection,
                                    const FieldDescriptor* field) {
    return true;
  }

  // An internal instance of the Tokenizer's error collector, used to
  // collect any base-level parse errors and feed them to the ParserImpl.
  class ParserErrorCollector : public io::ErrorCollector {
   public:
    explicit ParserErrorCollector(TemplateParser::Parser::ParserImpl* parser)
        : parser_(parser) {}

    virtual ~ParserErrorCollector() {}

    virtual void AddError(int line, int column, const std::string& message) {
      parser_->ReportError(line, column, message);
    }

    virtual void AddWarning(int line, int column, const std::string& message) {
      parser_->ReportWarning(line, column, message);
    }

   private:
    TemplateParser::Parser::ParserImpl* parser_;
  };

  io::ErrorCollector* error_collector_;
  const TextFormat::Finder* finder_;
  ParseInfoTree* parse_info_tree_;
  ParserErrorCollector tokenizer_error_collector_;
  Tokenizer tokenizer_;
  const Descriptor* root_message_type_;
  SingularOverwritePolicy singular_overwrite_policy_;
  const bool allow_case_insensitive_field_;
  const bool allow_unknown_field_;
  const bool allow_unknown_extension_;
  const bool allow_unknown_enum_;
  const bool allow_field_number_;
  const bool allow_partial_;
  int recursion_limit_;
  bool had_errors_;
};

namespace {

// Precedence for infix-style operators, matching C operator precedence.
const std::map<std::string, int>& InfixPrecedenceOrder() {
  static auto levels = new std::map<std::string, int>{
      {".", 1},  {"*", 3},   {"/", 3},  {"+", 4},  {"-", 4},
      {">", 6},  {"<", 6},   {">=", 6}, {"<=", 6}, {"==", 7},
      {"!=", 7}, {"&&", 11}, {"||", 12}};
  return *levels;
}

// Answers whether a token is an infix-style operator.
bool IsInfixOperator(const std::string& token) {
  return InfixPrecedenceOrder().count(token) > 0;
}

// A function-style operator, including a for or if expression.
bool IsFunctionOperator(const std::string& token) {
  static auto kTokens = new std::set<std::string>{
      "min",       "max",       "for",  "if",   "!",    "concat",
      "lowercase", "uppercase", "size", "dict", "list",
  };
  return kTokens->count(token) > 0;
}

// Merge all fields from a source Message into a destination Message.
// All nested Messages are constructed by the destination Message.
//
// This function is used to copy between a Message produced by the
// GeneratedMessageFactory ("template_rules_"), and a Message produced
// by the DynamicMessageFactory ("output").  These two Messages have
// different Descriptors so Message::MergeFrom cannot be applied directly,
// but they are expected to be equivalent.
absl::Status MergeFields(const Message& source, Message* dest) {
  std::unique_ptr<Message> temp(dest->New());
  std::string temp_str;
  RET_CHECK(TextFormat::PrintToString(source, &temp_str));
  RET_CHECK(TextFormat::ParseFromString(temp_str, temp.get()));
  dest->MergeFrom(*temp);
  return absl::OkStatus();
}

// Returns the (tag, index) pairs in a field path.
// For example, returns {{1, 1}, {2, 1}, {3, 1}} for path "/1[1]/2[1]/3[1]".
absl::Status ProtoPathSplit(const std::string& path,
                            ProtoUtilLite::ProtoPath* result) {
  absl::Status status;
  std::vector<std::string> ids = absl::StrSplit(path, '/');
  for (const std::string& id : ids) {
    if (id.length() > 0) {
      std::pair<std::string, std::string> id_pair =
          absl::StrSplit(id, absl::ByAnyChar("[]"));
      int tag = 0;
      int index = 0;
      bool ok = absl::SimpleAtoi(id_pair.first, &tag) &&
                absl::SimpleAtoi(id_pair.second, &index);
      if (!ok) {
        status.Update(absl::InvalidArgumentError(path));
      }
      result->push_back(std::make_pair(tag, index));
    }
  }
  return status;
}

// Serialize one field of a message.
void SerializeField(const Message* message, const FieldDescriptor* field,
                    std::vector<ProtoUtilLite::FieldValue>* result) {
  ProtoUtilLite::FieldValue message_bytes;
  CHECK(message->SerializePartialToString(&message_bytes));
  ProtoUtilLite::FieldAccess access(
      field->number(), static_cast<ProtoUtilLite::FieldType>(field->type()));
  MEDIAPIPE_CHECK_OK(access.SetMessage(message_bytes));
  *result = *access.mutable_field_values();
}

// For a non-repeated field, move the most recently parsed field value
// into the most recently parsed template expression.
void StowFieldValue(Message* message, TemplateExpression* expression) {
  const Reflection* reflection = message->GetReflection();
  const Descriptor* descriptor = message->GetDescriptor();
  ProtoUtilLite::ProtoPath path;
  MEDIAPIPE_CHECK_OK(ProtoPathSplit(expression->path(), &path));
  int field_number = path[path.size() - 1].first;
  const FieldDescriptor* field = descriptor->FindFieldByNumber(field_number);
  if (!field->is_repeated()) {
    std::vector<ProtoUtilLite::FieldValue> field_values;
    SerializeField(message, field, &field_values);
    *expression->mutable_field_value() = field_values[0];
    reflection->ClearField(message, field);
  }
}

// Strips first and last quotes from a string.
static void StripQuotes(std::string* str) {
  // Strip off the leading and trailing quotation marks from the value, if
  // there are any.
  if (str->size() > 1 && str->at(0) == str->at(str->size() - 1) &&
      (str->at(0) == '\'' || str->at(0) == '\"')) {
    str->erase(0, 1);
    str->erase(str->size() - 1);
  }
}

}  // namespace

class TemplateParser::Parser::MediaPipeParserImpl
    : public TemplateParser::Parser::ParserImpl {
  using TemplateParser::Parser::ParserImpl::ParserImpl;

  bool Parse(Message* output) override {
    // Parse protobufs into the output template "config" field.
    Message* config = output->GetReflection()->MutableMessage(
        output, output->GetDescriptor()->FindFieldByName("config"), nullptr);
    bool success = TemplateParser::Parser::ParserImpl::Parse(config);

    // Copy the template rules into the output template "rule" field.
    success &= MergeFields(template_rules_, output).ok();
    return success;
  }

 protected:
  void EnterField(const FieldDescriptor* field) override {
    RecordFieldPath(*field, parse_info_tree_->GetLastPath(field));
  }

  // Parse and record a template definition for the current field path.
  // The "base message" will be recorded at the field path as well.
  bool ConsumeFieldTemplate(Message* message) override {
    // find the current field path, including indices.
    // record the TemplateExpression at the path.
    TemplateExpression* expression = RecordTemplateRule();
    DO(Consume("%"));
    DO(ConsumeTemplateExpression(expression));
    DO(Consume("%"));
    // The %param% rule does not consume a field or an %end% tag.
    if (expression->op() == "param") {
      return true;
    }
    if (LookingAt("%")) {
      DO(ConsumeFieldTemplate(message));
    } else {
      DO(ConsumeField(message));
      StowFieldValue(message, expression);
    }
    DO(ConsumeEndTemplate());
    return true;
  }

  // Returns a placeholder value for the specified field.
  static void GetEmptyFieldValue(const FieldDescriptor* field,
                                 std::vector<ProtoUtilLite::FieldValue>* args) {
    auto field_type = static_cast<ProtoUtilLite::FieldType>(field->type());
    if (field_type == ProtoUtilLite::FieldType::TYPE_MESSAGE) {
      *args = {""};
    } else {
      MEDIAPIPE_CHECK_OK(ProtoUtilLite::Serialize({"1"}, field_type, args));
    }
  }

  // Inserts one value into the specified field.
  static void InsertFieldValue(
      Message* message, const FieldDescriptor* field,
      const std::vector<ProtoUtilLite::FieldValue>& args) {
    auto field_type = static_cast<ProtoUtilLite::FieldType>(field->type());
    ProtoUtilLite::FieldValue message_bytes;
    CHECK(message->SerializePartialToString(&message_bytes));
    MEDIAPIPE_CHECK_OK(ProtoUtilLite::ReplaceFieldRange(
        &message_bytes, {{field->number(), 0}}, 0, field_type, args));
    CHECK(message->ParsePartialFromString(message_bytes));
  }

  // Parse and record a template definition for the current field path.
  bool ConsumeValueTemplate(Message* message, const Reflection* reflection,
                            const FieldDescriptor* field) override {
    // Record a TemplateExpression with the current field path.
    TemplateExpression* expression = RecordTemplateRule();
    DO(Consume("%"));
    DO(ConsumeTemplateExpression(expression));
    DO(Consume("%"));
    RecordFieldPath(*field, parse_info_tree_->GetLastPath(field));

    // Leave a dummy value in place of the consumed field.
    std::vector<ProtoUtilLite::FieldValue> args;
    GetEmptyFieldValue(field, &args);
    InsertFieldValue(message, field, args);
    return true;
  }

  // Parse and record a template definition for the current field path.
  bool ConsumeMessageTemplate(Message* message, const Reflection* reflection,
                              const FieldDescriptor* field) override {
    // Record a TemplateExpression with the current message path.
    TemplateExpression* expression = RecordTemplateRule();
    DO(Consume("%"));
    DO(ConsumeTemplateExpression(expression));
    DO(Consume("%"));
    RecordFieldPath(*field, parse_info_tree_->GetPath());

    // Leave a dummy value in place of the consumed field.
    std::vector<ProtoUtilLite::FieldValue> args;
    GetEmptyFieldValue(field, &args);
    InsertFieldValue(message, field, args);
    return true;
  }

  // Parse %end%.
  bool ConsumeEndTemplate() {
    DO(Consume("%"));
    DO(Consume("end"));
    DO(Consume("%"));
    return true;
  }

  // Groups one infix operation according to operator precedence.
  // Groups the new rhs expression with previous rhs expressions if needed.
  void GroupOperator(const TemplateExpression& lhs, const std::string& op,
                     const TemplateExpression& rhs,
                     TemplateExpression* result) {
    if (IsInfixOperator(lhs.op()) &&
        InfixPrecedenceOrder().at(lhs.op()) > InfixPrecedenceOrder().at(op)) {
      result->set_op(lhs.op());
      (*result->add_arg()) = lhs.arg(0);
      GroupOperator(lhs.arg(1), op, rhs, result->add_arg());
    } else {
      result->set_op(op);
      (*result->add_arg()) = lhs;
      (*result->add_arg()) = rhs;
    }
  }

  // Parses a series of infix-style operations.
  bool ConsumeInfixExpression(TemplateExpression* result) {
    while (IsInfixOperator(tokenizer_.current().text)) {
      TemplateExpression lhs = *result;
      (*result) = TemplateExpression();
      std::string op = tokenizer_.current().text;
      tokenizer_.Next();
      TemplateExpression rhs;
      DO(ConsumePrefixExpression(&rhs));
      GroupOperator(lhs, op, rhs, result);
    }
    return true;
  }

  // Parses a template function-style operation.
  bool ConsumeFunctionExpression(TemplateExpression* result) {
    std::string function_name = tokenizer_.current().text;
    tokenizer_.Next();
    result->set_op(function_name);
    DO(Consume("("));
    bool success = true;
    while (true) {
      if (TryConsume(")")) {
        break;
      }
      if (!result->mutable_arg()->empty()) {
        success &= TryConsume(",") || TryConsume(":");
      }
      TemplateExpression arg;
      DO(ConsumeTemplateExpression(&arg));
      (*result->mutable_arg()->Add()) = arg;
    }
    return success;
  }

  // Parses a template parameter declaration.
  bool ConsumeParameterDeclaration(TemplateExpression* result) {
    DO(Consume("param"));
    result->set_op("param");
    std::string param_name;
    DO(ConsumeIdentifier(&param_name));
    result->add_arg()->set_param(param_name);
    if (TryConsume(":")) {
      DO(ConsumeTemplateExpression(result->add_arg()));
    }
    return true;
  }

  // Parses a template parameter reference.
  bool ConsumeParameterExpression(TemplateExpression* result) {
    std::string param_name;
    DO(ConsumeIdentifier(&param_name));
    result->set_param(param_name);
    return true;
  }

  // Parses a numeric or a string literal.
  bool ConsumeLiteral(TemplateExpression* result) {
    std::string token = tokenizer_.current().text;
    StripQuotes(&token);
    result->set_op("literal");
    result->set_param(token);
    tokenizer_.Next();
    return true;
  }

  // Parses a parenthesized expression.
  bool ConsumeGroupedExpression(TemplateExpression* result) {
    DO(Consume("("));
    result->set_op("paren");
    DO(ConsumeTemplateExpression(result->add_arg()));
    DO(Consume(")"));
    return true;
  }

  // Parses a TemplateExpression apart from infix operators.
  bool ConsumePrefixExpression(TemplateExpression* result) {
    if (LookingAt("(")) {
      return ConsumeGroupedExpression(result);
    }
    if (tokenizer_.current().text == "param") {
      return ConsumeParameterDeclaration(result);
    }
    if (IsFunctionOperator(tokenizer_.current().text)) {
      return ConsumeFunctionExpression(result);
    }
    if (LookingAtType(io::Tokenizer::TYPE_INTEGER) ||
        LookingAtType(io::Tokenizer::TYPE_FLOAT) ||
        LookingAtType(io::Tokenizer::TYPE_STRING)) {
      return ConsumeLiteral(result);
    }
    return ConsumeParameterExpression(result);
  }

  // Parses template parameter names and operators.
  bool ConsumeTemplateExpression(TemplateExpression* result) {
    bool success = ConsumePrefixExpression(result);
    if (IsInfixOperator(tokenizer_.current().text)) {
      return ConsumeInfixExpression(result);
    }
    return success;
  }

  // Records a template expression for the current field-path.
  TemplateExpression* RecordTemplateRule() {
    return template_rules_.mutable_rule()->Add();
  }

  // Records the field path and field type for the rule or rules targeting
  // a certain field.
  void RecordFieldPath(const FieldDescriptor& field, const std::string& path) {
    for (int i = template_rules_.rule().size() - 1; i >= 0; --i) {
      auto rule = template_rules_.mutable_rule()->Mutable(i);
      if (rule->has_path() || rule->op() == "param") {
        break;
      }
      rule->set_path(path);
      rule->set_field_type(
          static_cast<mediapipe::FieldDescriptorProto::Type>(field.type()));
    }
  }

  mediapipe::CalculatorGraphTemplate template_rules_;
};

#undef DO

// ===========================================================================

TemplateParser::Parser::Parser()
    : error_collector_(nullptr),
      finder_(nullptr),
      parse_info_tree_(nullptr),
      allow_partial_(false),
      allow_case_insensitive_field_(false),
      allow_unknown_field_(false),
      allow_unknown_enum_(false),
      allow_field_number_(false),
      allow_relaxed_whitespace_(false),
      allow_singular_overwrites_(false) {
  parse_info_tree_ = new ParseInfoTree();
}

TemplateParser::Parser::~Parser() { delete parse_info_tree_; }

bool TemplateParser::Parser::Parse(io::ZeroCopyInputStream* input,
                                   Message* output) {
  output->Clear();

  ParserImpl::SingularOverwritePolicy overwrites_policy =
      allow_singular_overwrites_ ? ParserImpl::ALLOW_SINGULAR_OVERWRITES
                                 : ParserImpl::FORBID_SINGULAR_OVERWRITES;

  int recursion_limit = std::numeric_limits<int>::max();
  bool allow_unknown_extension = false;
  MediaPipeParserImpl parser(
      output->GetDescriptor(), input, error_collector_, finder_,
      parse_info_tree_, overwrites_policy, allow_case_insensitive_field_,
      allow_unknown_field_, allow_unknown_extension, allow_unknown_enum_,
      allow_field_number_, allow_relaxed_whitespace_, allow_partial_,
      recursion_limit);
  return MergeUsingImpl(input, output, &parser);
}

bool TemplateParser::Parser::ParseFromString(const std::string& input,
                                             Message* output) {
  io::ArrayInputStream input_stream(input.data(), input.size());
  return Parse(&input_stream, output);
}

bool TemplateParser::Parser::Merge(io::ZeroCopyInputStream* input,
                                   Message* output) {
  int recursion_limit = std::numeric_limits<int>::max();
  bool allow_unknown_extension = false;
  MediaPipeParserImpl parser(
      output->GetDescriptor(), input, error_collector_, finder_,
      parse_info_tree_, ParserImpl::ALLOW_SINGULAR_OVERWRITES,
      allow_case_insensitive_field_, allow_unknown_field_,
      allow_unknown_extension, allow_unknown_enum_, allow_field_number_,
      allow_relaxed_whitespace_, allow_partial_, recursion_limit);
  return MergeUsingImpl(input, output, &parser);
}

bool TemplateParser::Parser::MergeFromString(const std::string& input,
                                             Message* output) {
  io::ArrayInputStream input_stream(input.data(), input.size());
  return Merge(&input_stream, output);
}

bool TemplateParser::Parser::MergeUsingImpl(
    io::ZeroCopyInputStream* /* input */, Message* output,
    ParserImpl* parser_impl) {
  if (!parser_impl->Parse(output)) return false;
  if (!allow_partial_ && !output->IsInitialized()) {
    std::vector<std::string> missing_fields;
    output->FindInitializationErrors(&missing_fields);
    parser_impl->ReportError(-1, 0,
                             "Message missing required fields: " +
                                 absl::StrJoin(missing_fields, ", "));
    return false;
  }
  return true;
}

bool TemplateParser::Parser::ParseFieldValueFromString(
    const std::string& input, const FieldDescriptor* field, Message* output) {
  io::ArrayInputStream input_stream(input.data(), input.size());
  int recursion_limit = std::numeric_limits<int>::max();
  bool allow_unknown_extension = false;
  ParserImpl parser(
      output->GetDescriptor(), &input_stream, error_collector_, finder_,
      parse_info_tree_, ParserImpl::ALLOW_SINGULAR_OVERWRITES,
      allow_case_insensitive_field_, allow_unknown_field_,
      allow_unknown_extension, allow_unknown_enum_, allow_field_number_,
      allow_relaxed_whitespace_, allow_partial_, recursion_limit);
  return parser.ParseField(field, output);
}

}  // namespace tool
}  // namespace mediapipe
