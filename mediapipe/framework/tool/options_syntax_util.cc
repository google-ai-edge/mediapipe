#include "mediapipe/framework/tool/options_syntax_util.h"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/any_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/name_util.h"
#include "mediapipe/framework/tool/options_registry.h"

namespace mediapipe {
namespace tool {

namespace {

// StrSplit Delimiter to split strings at single colon tokens, ignoring
// double-colon tokens.
class SingleColonDelimiter {
 public:
  SingleColonDelimiter() {}
  absl::string_view Find(absl::string_view text, size_t pos) const {
    while (pos < text.length()) {
      size_t p = text.find(':', pos);
      p = (p == absl::string_view::npos) ? text.length() : p;
      if (p >= text.length() - 1 || text[p + 1] != ':') {
        return text.substr(p, 1);
      }
      pos = p + 2;
    }
    return text.substr(text.length(), 0);
  }
};

}  // namespace

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

  // Return the extension-type specified for an option field.
  absl::string_view ExtensionType(absl::string_view option_name) {
    constexpr absl::string_view kExt = "Ext::";
    if (absl::StartsWithIgnoreCase(option_name, kExt)) {
      return option_name.substr(kExt.size());
    }
    return "";
  }

  // Returns the field names encoded in an options tag.
  std::vector<absl::string_view> OptionTagNames(absl::string_view tag) {
    if (absl::StartsWith(tag, syntax_.tag_name)) {
      tag = tag.substr(syntax_.tag_name.length());
    } else if (absl::StartsWith(tag, syntax_.packet_name)) {
      tag = tag.substr(syntax_.packet_name.length());
    }
    if (absl::StartsWith(tag, syntax_.separator)) {
      tag = tag.substr(syntax_.separator.length());
    }
    return absl::StrSplit(tag, syntax_.separator);
  }

  // Returns the field-path for an option stream-tag.
  FieldPath OptionFieldPath(absl::string_view tag,
                            const Descriptor* descriptor) {
    std::vector<absl::string_view> name_tags = OptionTagNames(tag);
    FieldPath result;
    for (absl::string_view name_tag : name_tags) {
      if (name_tag.empty()) {
        continue;
      }
      absl::string_view option_name = OptionFieldName(name_tag);
      int index;
      if (absl::SimpleAtoi(option_name, &index)) {
        result.back().index = index;
      } else if (!ExtensionType(option_name).empty()) {
        std::string extension_type = std::string(ExtensionType(option_name));
        result.push_back({nullptr, 0, extension_type});
        descriptor = OptionsRegistry::GetProtobufDescriptor(extension_type);
      } else {
        if (descriptor == nullptr) {
          break;
        }
        auto field = descriptor->FindFieldByName(std::string(option_name));
        descriptor = field ? field->message_type() : nullptr;
        result.push_back({std::move(field), -1});
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
  std::string OptionFieldsTag(absl::string_view option_names) {
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

std::string OptionsSyntaxUtil::OptionFieldsTag(absl::string_view option_names) {
  return syntax_helper_->OptionFieldsTag(option_names);
}

OptionsSyntaxUtil::FieldPath OptionsSyntaxUtil::OptionFieldPath(
    absl::string_view tag, const Descriptor* descriptor) {
  return syntax_helper_->OptionFieldPath(tag, descriptor);
}
std::vector<absl::string_view> OptionsSyntaxUtil::StrSplitTags(
    absl::string_view tag_and_name) {
  return absl::StrSplit(tag_and_name, SingleColonDelimiter());
}

}  // namespace tool
}  // namespace mediapipe
