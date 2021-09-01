#include "mediapipe/framework/tool/options_syntax_util.h"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/any_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/name_util.h"

namespace mediapipe {
namespace tool {

// Helper functions for parsing the graph options syntax.
class OptionsSyntaxUtil::OptionsSyntaxHelper {
 public:
  // The usual graph options syntax tokens.
  OptionsSyntaxHelper() : syntax_{"OPTIONS", "options", "/"} {}

  // Returns the tag name for an option protobuf field.
  std::string OptionFieldTag(const std::string& name) { return name; }

  // Returns the packet name for an option protobuf field.
  absl::string_view OptionFieldPacket(absl::string_view name) { return name; }

  // Returns the option protobuf field name for a tag or packet name.
  absl::string_view OptionFieldName(absl::string_view name) { return name; }

  // Returns the field-path for an option stream-tag.
  FieldPath OptionFieldPath(const std::string& tag,
                            const Descriptor* descriptor) {
    int prefix = syntax_.tag_name.length() + syntax_.separator.length();
    std::string suffix = tag.substr(prefix);
    std::vector<absl::string_view> name_tags =
        absl::StrSplit(suffix, syntax_.separator);
    FieldPath result;
    for (absl::string_view name_tag : name_tags) {
      if (name_tag.empty()) {
        continue;
      }
      absl::string_view option_name = OptionFieldName(name_tag);
      int index;
      if (absl::SimpleAtoi(option_name, &index)) {
        result.back().second = index;
      } else {
        auto field = descriptor->FindFieldByName(std::string(option_name));
        descriptor = field ? field->message_type() : nullptr;
        result.push_back({std::move(field), 0});
      }
    }
    return result;
  }

  // Returns the option field name for a graph options packet name.
  std::string GraphOptionFieldName(const std::string& graph_option_name) {
    int prefix = syntax_.packet_name.length() + syntax_.separator.length();
    std::string result = graph_option_name;
    result.erase(0, prefix);
    return result;
  }

  // Returns the graph options packet name for an option field name.
  std::string GraphOptionName(const std::string& option_name) {
    std::string packet_prefix =
        syntax_.packet_name + absl::AsciiStrToLower(syntax_.separator);
    return absl::StrCat(packet_prefix, option_name);
  }

  // Returns the tag name for a graph option.
  std::string OptionTagName(const std::string& option_name) {
    return absl::StrCat(syntax_.tag_name, syntax_.separator,
                        OptionFieldTag(option_name));
  }

  // Converts slash-separated field names into a tag name.
  std::string OptionFieldsTag(const std::string& option_names) {
    std::string tag_prefix = syntax_.tag_name + syntax_.separator;
    std::vector<absl::string_view> names = absl::StrSplit(option_names, '/');
    if (!names.empty() && names[0] == syntax_.tag_name) {
      names.erase(names.begin());
    }
    if (!names.empty() && names[0] == syntax_.packet_name) {
      names.erase(names.begin());
    }
    std::string result;
    std::string sep = "";
    for (absl::string_view v : names) {
      absl::StrAppend(&result, sep, OptionFieldTag(std::string(v)));
      sep = syntax_.separator;
    }
    result = tag_prefix + result;
    return result;
  }

  // Token definitions for the graph options syntax.
  struct OptionsSyntax {
    // The tag name for an options protobuf.
    std::string tag_name;
    // The packet name for an options protobuf.
    std::string packet_name;
    // The separator between nested options fields.
    std::string separator;
  };

  OptionsSyntax syntax_;
};  // class OptionsSyntaxHelper

OptionsSyntaxUtil::OptionsSyntaxUtil()
    : syntax_helper_(std::make_unique<OptionsSyntaxHelper>()) {}

OptionsSyntaxUtil::OptionsSyntaxUtil(const std::string& tag_name)
    : OptionsSyntaxUtil() {
  syntax_helper_->syntax_.tag_name = tag_name;
}

OptionsSyntaxUtil::OptionsSyntaxUtil(const std::string& tag_name,
                                     const std::string& packet_name,
                                     const std::string& separator)
    : OptionsSyntaxUtil() {
  syntax_helper_->syntax_.tag_name = tag_name;
  syntax_helper_->syntax_.packet_name = packet_name;
  syntax_helper_->syntax_.separator = separator;
}

OptionsSyntaxUtil::~OptionsSyntaxUtil() {}

std::string OptionsSyntaxUtil::OptionFieldsTag(
    const std::string& option_names) {
  return syntax_helper_->OptionFieldsTag(option_names);
}

OptionsSyntaxUtil::FieldPath OptionsSyntaxUtil::OptionFieldPath(
    const std::string& tag, const Descriptor* descriptor) {
  return syntax_helper_->OptionFieldPath(tag, descriptor);
}

}  // namespace tool
}  // namespace mediapipe
