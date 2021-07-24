
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/logging.h"

ABSL_FLAG(std::string, input_path, "",
          "Full path of the FileDescriptorSet to summarize. ");
ABSL_FLAG(std::string, root_type_name_output_path, "",
          "Where to write the output root message type name. ");
ABSL_FLAG(std::string, root_type_macro_output_path, "",
          "Where to write the output root message type macro. ");

namespace mediapipe {

using proto_ns::DescriptorProto;
using proto_ns::FileDescriptorProto;
using proto_ns::FileDescriptorSet;

// Utility to extract summary data about protobuf descriptors.
//
// This utility is currently used by the build rule mediapipe_options_library()
// to recover the package-name and type-name associated with each
// mediapipe_proto_library() target.
class DescriptorReader {
 public:
  // Returns a FileDescriptor that is not referenced by other FileDescriptors
  // in a FileDescriptorSet.
  static FileDescriptorProto FindTopFile(const FileDescriptorSet& files) {
    std::set<std::string> file_names;
    for (const FileDescriptorProto& file : files.file()) {
      file_names.insert(file.name());
    }
    for (const FileDescriptorProto& file : files.file()) {
      for (const std::string& dep : file.dependency()) {
        file_names.erase(dep);
      }
    }
    for (const FileDescriptorProto& file : files.file()) {
      if (file_names.count(file.name()) > 0) {
        return file;
      }
    }
    return FileDescriptorProto();
  }

  static std::string CleanTypeName(const std::string& type_name) {
    return (type_name.rfind('.', 0) == 0) ? type_name.substr(1) : type_name;
  }

  static std::string CleanTypeName(const std::string& package,
                                   const std::string& name) {
    return absl::StrCat(package, ".", name);
  }

  // Returns the length of the common prefix between two strings.
  static int MatchingPrefixLength(const std::string& s, const std::string& t) {
    int i = 0;
    while (i < std::min(s.size(), t.size()) && s[i] == t[i]) {
      ++i;
    }
    return i;
  }

  // Returns the type-name that best matches the descriptor file-name.
  static std::string BestTypeName(const std::set<std::string>& type_names,
                                  const FileDescriptorProto& file) {
    std::string proto_name = std::string(file::Basename(file.name()));
    proto_name = proto_name.substr(
        0, proto_name.size() - file::Extension(proto_name).size() - 1);
    proto_name.erase(std::remove(proto_name.begin(), proto_name.end(), '_'),
                     proto_name.end());
    std::string result = "";
    int best_match = -1;
    for (const std::string& type_name : type_names) {
      std::string name = absl::AsciiStrToLower(type_name);
      if (name.rfind('.') != std::string::npos) {
        name = name.substr(name.rfind('.') + 1);
      }
      int m = MatchingPrefixLength(proto_name, name);
      if (m > best_match) {
        best_match = m;
        result = type_name;
      }
    }
    return result;
  }

  // Returns a DescriptorProto that is not referenced by other DescriptorProtos
  // in a FileDescriptorProto.
  static DescriptorProto FindTopDescriptor(const FileDescriptorProto& file) {
    std::set<std::string> type_names;
    std::set<std::string> refs;
    for (const DescriptorProto& descriptor : file.message_type()) {
      type_names.insert(CleanTypeName(file.package(), descriptor.name()));
    }
    std::string best_name = BestTypeName(type_names, file);
    for (const DescriptorProto& descriptor : file.message_type()) {
      if (best_name == CleanTypeName(file.package(), descriptor.name())) {
        return descriptor;
      }
    }
    return DescriptorProto();
  }

  static std::string FindTopTypeName(const FileDescriptorSet& files) {
    FileDescriptorProto file = FindTopFile(files);
    DescriptorProto descriptor = FindTopDescriptor(file);
    return CleanTypeName(file.package(), descriptor.name());
  }

  static FileDescriptorSet ReadFileDescriptorSet(const std::string& path) {
    std::string contents;
    CHECK_OK(file::GetContents(path, &contents));
    proto_ns::FileDescriptorSet result;
    result.ParseFromString(contents);
    return result;
  }

  static void WriteFile(const std::string& path, const std::string& contents) {
    CHECK_OK(file::SetContents(path, contents));
  }

  static void WriteMessageTypeName(const std::string& path,
                                   const FileDescriptorSet& files) {
    FileDescriptorProto file = FindTopFile(files);
    DescriptorProto descriptor = FindTopDescriptor(file);
    std::string type_name = mediapipe::DescriptorReader::FindTopTypeName(files);
    mediapipe::DescriptorReader::WriteFile(
        absl::GetFlag(FLAGS_root_type_name_output_path), type_name);
  }

  static void WriteMessageTypeMacro(const std::string& path,
                                    const FileDescriptorSet& files) {
    FileDescriptorProto file = FindTopFile(files);
    DescriptorProto descriptor = FindTopDescriptor(file);
    std::string type_package =
        absl::StrReplaceAll(file.package(), {{".", "::"}});
    std::string type_name = descriptor.name();
    std::string contents =
        absl::StrCat("#define MP_OPTION_TYPE_NS ", type_package, "\n") +
        absl::StrCat("#define MP_OPTION_TYPE_NAME ", type_name, "\n");
    WriteFile(path, contents);
  }
};

}  // namespace mediapipe

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  auto files = mediapipe::DescriptorReader::ReadFileDescriptorSet(
      absl::GetFlag(FLAGS_input_path));
  if (!absl::GetFlag(FLAGS_root_type_name_output_path).empty()) {
    mediapipe::DescriptorReader::WriteMessageTypeName(
        absl::GetFlag(FLAGS_root_type_name_output_path), files);
  }
  if (!absl::GetFlag(FLAGS_root_type_macro_output_path).empty()) {
    mediapipe::DescriptorReader::WriteMessageTypeMacro(
        absl::GetFlag(FLAGS_root_type_macro_output_path), files);
  }
  return EXIT_SUCCESS;
}
