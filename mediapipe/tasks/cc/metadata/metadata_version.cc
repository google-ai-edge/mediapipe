/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "mediapipe/tasks/cc/metadata/metadata_version.h"

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "flatbuffers/flatbuffers.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/tools/logging.h"

namespace mediapipe {
namespace tasks {
namespace metadata {
namespace {

using ::tflite::AssociatedFileType_SCANN_INDEX_FILE;
using ::tflite::AssociatedFileType_VOCABULARY;
using ::tflite::ContentProperties_AudioProperties;
using ::tflite::GetModelMetadata;
using ::tflite::ProcessUnitOptions_BertTokenizerOptions;
using ::tflite::ProcessUnitOptions_RegexTokenizerOptions;
using ::tflite::ProcessUnitOptions_SentencePieceTokenizerOptions;

// Members that are added to the metadata schema after the initial version
// of 1.0.0.
enum class SchemaMembers {
  kAssociatedFileTypeVocabulary = 0,
  kSubGraphMetadataInputProcessUnits = 1,
  kSubGraphMetadataOutputProcessUnits = 2,
  kProcessUnitOptionsBertTokenizerOptions = 3,
  kProcessUnitOptionsSentencePieceTokenizerOptions = 4,
  kSubGraphMetadataInputTensorGroups = 5,
  kSubGraphMetadataOutputTensorGroups = 6,
  kProcessUnitOptionsRegexTokenizerOptions = 7,
  kContentPropertiesAudioProperties = 8,
  kAssociatedFileTypeScannIndexFile = 9,
  kAssociatedFileVersion = 10,
};

// Helper class to compare semantic versions in terms of three integers, major,
// minor, and patch.
class Version {
 public:
  explicit Version(int major, int minor = 0, int patch = 0)
      : version_({major, minor, patch}) {}

  explicit Version(const std::string& version) {
    const std::vector<std::string> vec = absl::StrSplit(version, '.');
    // The version string should always be less than four numbers.
    TFLITE_DCHECK(vec.size() <= kElementNumber && !vec.empty());
    version_[0] = std::stoi(vec[0]);
    version_[1] = vec.size() > 1 ? std::stoi(vec[1]) : 0;
    version_[2] = vec.size() > 2 ? std::stoi(vec[2]) : 0;
  }

  // Compares two semantic version numbers.
  //
  // Example results when comparing two versions strings:
  //   "1.9" precedes "1.14";
  //   "1.14" precedes "1.14.1";
  //   "1.14" and "1.14.0" are equal.
  //
  // Returns the value 0 if the two versions are equal; a value less than 0 if
  // *this precedes v; a value greater than 0 if v precedes *this.
  int Compare(const Version& v) {
    for (int i = 0; i < kElementNumber; ++i) {
      if (version_[i] != v.version_[i]) {
        return version_[i] < v.version_[i] ? -1 : 1;
      }
    }
    return 0;
  }

  // Converts version_ into a version string.
  std::string ToString() { return absl::StrJoin(version_, "."); }

 private:
  static constexpr int kElementNumber = 3;
  std::array<int, kElementNumber> version_;
};

Version GetMemberVersion(SchemaMembers member) {
  switch (member) {
    case SchemaMembers::kAssociatedFileTypeVocabulary:
      return Version(1, 0, 1);
    case SchemaMembers::kSubGraphMetadataInputProcessUnits:
      return Version(1, 1, 0);
    case SchemaMembers::kSubGraphMetadataOutputProcessUnits:
      return Version(1, 1, 0);
    case SchemaMembers::kProcessUnitOptionsBertTokenizerOptions:
      return Version(1, 1, 0);
    case SchemaMembers::kProcessUnitOptionsSentencePieceTokenizerOptions:
      return Version(1, 1, 0);
    case SchemaMembers::kSubGraphMetadataInputTensorGroups:
      return Version(1, 2, 0);
    case SchemaMembers::kSubGraphMetadataOutputTensorGroups:
      return Version(1, 2, 0);
    case SchemaMembers::kProcessUnitOptionsRegexTokenizerOptions:
      return Version(1, 2, 1);
    case SchemaMembers::kContentPropertiesAudioProperties:
      return Version(1, 3, 0);
    case SchemaMembers::kAssociatedFileTypeScannIndexFile:
      return Version(1, 4, 0);
    case SchemaMembers::kAssociatedFileVersion:
      return Version(1, 4, 1);
    default:
      // Should never happen.
      TFLITE_LOG(FATAL) << "Unsupported schema member: "
                        << static_cast<int>(member);
  }
  // Should never happen.
  return Version(0, 0, 0);
}

// Updates min_version if it precedes the new_version.
inline void UpdateMinimumVersion(const Version& new_version,
                                 Version* min_version) {
  if (min_version->Compare(new_version) < 0) {
    *min_version = new_version;
  }
}

template <typename T>
void UpdateMinimumVersionForTable(const T* table, Version* min_version);

template <typename T>
void UpdateMinimumVersionForArray(
    const flatbuffers::Vector<flatbuffers::Offset<T>>* array,
    Version* min_version) {
  if (array == nullptr) return;

  for (int i = 0; i < array->size(); ++i) {
    UpdateMinimumVersionForTable<T>(array->Get(i), min_version);
  }
}

template <>
void UpdateMinimumVersionForTable<tflite::AssociatedFile>(
    const tflite::AssociatedFile* table, Version* min_version) {
  if (table == nullptr) return;

  if (table->type() == AssociatedFileType_VOCABULARY) {
    UpdateMinimumVersion(
        GetMemberVersion(SchemaMembers::kAssociatedFileTypeVocabulary),
        min_version);
  }

  if (table->type() == AssociatedFileType_SCANN_INDEX_FILE) {
    UpdateMinimumVersion(
        GetMemberVersion(SchemaMembers::kAssociatedFileTypeScannIndexFile),
        min_version);
  }

  if (table->version() != nullptr) {
    UpdateMinimumVersion(
        GetMemberVersion(SchemaMembers::kAssociatedFileVersion), min_version);
  }
}

template <>
void UpdateMinimumVersionForTable<tflite::ProcessUnit>(
    const tflite::ProcessUnit* table, Version* min_version) {
  if (table == nullptr) return;

  tflite::ProcessUnitOptions process_unit_type = table->options_type();
  if (process_unit_type == ProcessUnitOptions_BertTokenizerOptions) {
    UpdateMinimumVersion(
        GetMemberVersion(
            SchemaMembers::kProcessUnitOptionsBertTokenizerOptions),
        min_version);
  }
  if (process_unit_type == ProcessUnitOptions_SentencePieceTokenizerOptions) {
    UpdateMinimumVersion(
        GetMemberVersion(
            SchemaMembers::kProcessUnitOptionsSentencePieceTokenizerOptions),
        min_version);
  }
  if (process_unit_type == ProcessUnitOptions_RegexTokenizerOptions) {
    UpdateMinimumVersion(
        GetMemberVersion(
            SchemaMembers::kProcessUnitOptionsRegexTokenizerOptions),
        min_version);
  }
}

template <>
void UpdateMinimumVersionForTable<tflite::Content>(const tflite::Content* table,
                                                   Version* min_version) {
  if (table == nullptr) return;

  // Checks the ContenProperties field.
  if (table->content_properties_type() == ContentProperties_AudioProperties) {
    UpdateMinimumVersion(
        GetMemberVersion(SchemaMembers::kContentPropertiesAudioProperties),
        min_version);
  }
}

template <>
void UpdateMinimumVersionForTable<tflite::TensorMetadata>(
    const tflite::TensorMetadata* table, Version* min_version) {
  if (table == nullptr) return;

  // Checks the associated_files field.
  UpdateMinimumVersionForArray<tflite::AssociatedFile>(
      table->associated_files(), min_version);

  // Checks the process_units field.
  UpdateMinimumVersionForArray<tflite::ProcessUnit>(table->process_units(),
                                                    min_version);

  // Check the content field.
  UpdateMinimumVersionForTable<tflite::Content>(table->content(), min_version);
}

template <>
void UpdateMinimumVersionForTable<tflite::SubGraphMetadata>(
    const tflite::SubGraphMetadata* table, Version* min_version) {
  if (table == nullptr) return;

  // Checks in the input/output metadata arrays.
  UpdateMinimumVersionForArray<tflite::TensorMetadata>(
      table->input_tensor_metadata(), min_version);
  UpdateMinimumVersionForArray<tflite::TensorMetadata>(
      table->output_tensor_metadata(), min_version);

  // Checks the associated_files field.
  UpdateMinimumVersionForArray<tflite::AssociatedFile>(
      table->associated_files(), min_version);

  // Checks for the input_process_units field.
  if (table->input_process_units() != nullptr) {
    UpdateMinimumVersion(
        GetMemberVersion(SchemaMembers::kSubGraphMetadataInputProcessUnits),
        min_version);
    UpdateMinimumVersionForArray<tflite::ProcessUnit>(
        table->input_process_units(), min_version);
  }

  // Checks for the output_process_units field.
  if (table->output_process_units() != nullptr) {
    UpdateMinimumVersion(
        GetMemberVersion(SchemaMembers::kSubGraphMetadataOutputProcessUnits),
        min_version);
    UpdateMinimumVersionForArray<tflite::ProcessUnit>(
        table->output_process_units(), min_version);
  }

  // Checks for the input_tensor_groups field.
  if (table->input_tensor_groups() != nullptr) {
    UpdateMinimumVersion(
        GetMemberVersion(SchemaMembers::kSubGraphMetadataInputTensorGroups),
        min_version);
  }

  // Checks for the output_tensor_groups field.
  if (table->output_tensor_groups() != nullptr) {
    UpdateMinimumVersion(
        GetMemberVersion(SchemaMembers::kSubGraphMetadataOutputTensorGroups),
        min_version);
  }
}

template <>
void UpdateMinimumVersionForTable<tflite::ModelMetadata>(
    const tflite::ModelMetadata* table, Version* min_version) {
  if (table == nullptr) {
    // Should never happen, because VerifyModelMetadataBuffer has verified it.
    TFLITE_LOG(FATAL) << "The ModelMetadata object is null.";
    return;
  }

  // Checks the subgraph_metadata field.
  if (table->subgraph_metadata() != nullptr) {
    for (int i = 0; i < table->subgraph_metadata()->size(); ++i) {
      UpdateMinimumVersionForTable<tflite::SubGraphMetadata>(
          table->subgraph_metadata()->Get(i), min_version);
    }
  }

  // Checks the associated_files field.
  UpdateMinimumVersionForArray<tflite::AssociatedFile>(
      table->associated_files(), min_version);
}

}  // namespace

TfLiteStatus GetMinimumMetadataParserVersion(const uint8_t* buffer_data,
                                             size_t buffer_size,
                                             std::string* min_version_str) {
  flatbuffers::Verifier verifier =
      flatbuffers::Verifier(buffer_data, buffer_size);
  if (!tflite::VerifyModelMetadataBuffer(verifier)) {
    TFLITE_LOG(ERROR) << "The model metadata is not a valid FlatBuffer buffer.";
    return kTfLiteError;
  }

  static constexpr char kDefaultVersion[] = "1.0.0";
  Version min_version = Version(kDefaultVersion);

  // Checks if any member declared after 1.0.0 (such as those in
  // SchemaMembers) exists, and updates min_version accordingly. The minimum
  // metadata parser version will be the largest version number of all fields
  // that has been added to a metadata flatbuffer
  const tflite::ModelMetadata* model_metadata = GetModelMetadata(buffer_data);

  // All tables in the metadata schema should have their dedicated
  // UpdateMinimumVersionForTable<Foo>() methods, respectively. We'll gradually
  // add these methods when new fields show up in later schema versions.
  //
  // UpdateMinimumVersionForTable<Foo>() takes a const pointer of Foo. The
  // pointer can be a nullptr if Foo is not populated into the corresponding
  // table of the Flatbuffer object. In this case,
  // UpdateMinimumVersionFor<Foo>() will be skipped. An exception is
  // UpdateMinimumVersionForModelMetadata(), where ModelMetadata is the root
  // table, and it won't be null.
  UpdateMinimumVersionForTable<tflite::ModelMetadata>(model_metadata,
                                                      &min_version);

  *min_version_str = min_version.ToString();
  return kTfLiteOk;
}

}  // namespace metadata
}  // namespace tasks
}  // namespace mediapipe
