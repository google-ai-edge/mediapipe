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

#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"

#include <string>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "contrib/minizip/ioapi.h"
#include "contrib/minizip/unzip.h"
#include "flatbuffers/flatbuffers.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/metadata/utils/zip_readonly_mem_file.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace metadata {

namespace {
constexpr char kMetadataBufferName[] = "TFLITE_METADATA";

using ::absl::StatusCode;
using ::flatbuffers::Offset;
using ::flatbuffers::Vector;
using ::mediapipe::tasks::CreateStatusWithPayload;
using ::mediapipe::tasks::MediaPipeTasksStatus;
using ::tflite::TensorMetadata;

// Util to get item from src_vector specified by index.
template <typename T>
const T* GetItemFromVector(
    const flatbuffers::Vector<flatbuffers::Offset<T>>* src_vector, int index) {
  if (src_vector == nullptr || index < 0 || index >= src_vector->size()) {
    return nullptr;
  }
  return src_vector->Get(index);
}

// Wrapper function around calls to unzip to avoid repeating conversion logic
// from error code to Status.
absl::Status UnzipErrorToStatus(int error) {
  if (error != UNZ_OK) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Unable to read associated file in zip archive.",
        MediaPipeTasksStatus::kMetadataAssociatedFileZipError);
  }
  return absl::OkStatus();
}

// Stores a file name, position in zip buffer and size.
struct ZipFileInfo {
  std::string name;
  ZPOS64_T position;
  ZPOS64_T size;
};

// Returns the ZipFileInfo corresponding to the current file in the provided
// unzFile object.
absl::StatusOr<ZipFileInfo> GetCurrentZipFileInfo(const unzFile& zf) {
  // Open file in raw mode, as data is expected to be uncompressed.
  int method;
  MP_RETURN_IF_ERROR(UnzipErrorToStatus(
      unzOpenCurrentFile2(zf, &method, /*level=*/nullptr, /*raw=*/1)));
  if (method != Z_NO_COMPRESSION) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Expected uncompressed zip archive.",
        MediaPipeTasksStatus::kMetadataAssociatedFileZipError);
  }

  // Get file info a first time to get filename size.
  unz_file_info64 file_info;
  MP_RETURN_IF_ERROR(UnzipErrorToStatus(unzGetCurrentFileInfo64(
      zf, &file_info, /*szFileName=*/nullptr, /*szFileNameBufferSize=*/0,
      /*extraField=*/nullptr, /*extraFieldBufferSize=*/0,
      /*szComment=*/nullptr, /*szCommentBufferSize=*/0)));

  // Second call to get file name.
  auto file_name_size = file_info.size_filename;
  char* c_file_name = (char*)malloc(file_name_size);
  MP_RETURN_IF_ERROR(UnzipErrorToStatus(unzGetCurrentFileInfo64(
      zf, &file_info, c_file_name, file_name_size,
      /*extraField=*/nullptr, /*extraFieldBufferSize=*/0,
      /*szComment=*/nullptr, /*szCommentBufferSize=*/0)));
  std::string file_name = std::string(c_file_name, file_name_size);
  free(c_file_name);

  // Get position in file.
  auto position = unzGetCurrentFileZStreamPos64(zf);
  if (position == 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Unable to read file in zip archive.",
        MediaPipeTasksStatus::kMetadataAssociatedFileZipError);
  }

  // Close file and return.
  MP_RETURN_IF_ERROR(UnzipErrorToStatus(unzCloseCurrentFile(zf)));

  ZipFileInfo result{};
  result.name = file_name;
  result.position = position;
  result.size = file_info.uncompressed_size;
  return result;
}
}  // namespace

/* static */
absl::StatusOr<std::unique_ptr<ModelMetadataExtractor>>
ModelMetadataExtractor::CreateFromModelBuffer(const char* buffer_data,
                                              size_t buffer_size) {
  // Use absl::WrapUnique() to call private constructor:
  // https://abseil.io/tips/126.
  std::unique_ptr<ModelMetadataExtractor> extractor =
      absl::WrapUnique(new ModelMetadataExtractor());
  MP_RETURN_IF_ERROR(extractor->InitFromModelBuffer(buffer_data, buffer_size));
  return extractor;
}

/* static */
absl::StatusOr<const tflite::ProcessUnit*>
ModelMetadataExtractor::FindFirstProcessUnit(
    const tflite::TensorMetadata& tensor_metadata,
    tflite::ProcessUnitOptions type) {
  const tflite::ProcessUnit* result = nullptr;
  if (tensor_metadata.process_units() == nullptr) {
    return result;
  }
  for (const auto process_unit : *tensor_metadata.process_units()) {
    if (process_unit->options_type() == type) {
      if (result != nullptr) {
        return CreateStatusWithPayload(
            StatusCode::kInvalidArgument,
            absl::StrCat("Found multiple ProcessUnits with type=",
                         tflite::EnumNameProcessUnitOptions(type),
                         ", expected at most one."),
            MediaPipeTasksStatus::kMetadataInvalidProcessUnitsError);
      }
      result = process_unit;
    }
  }
  return result;
}

/* static */
std::string ModelMetadataExtractor::FindFirstAssociatedFileName(
    const tflite::TensorMetadata& tensor_metadata,
    tflite::AssociatedFileType type, absl::string_view locale) {
  if (tensor_metadata.associated_files() == nullptr) {
    return std::string();
  }
  for (const auto associated_file : *tensor_metadata.associated_files()) {
    if (associated_file->type() != type || associated_file->name() == nullptr) {
      continue;
    }
    if (locale.empty() || (associated_file->locale() != nullptr &&
                           locale == associated_file->locale()->str())) {
      return associated_file->name()->str();
    }
  }
  return std::string();
}

absl::Status ModelMetadataExtractor::InitFromModelBuffer(
    const char* buffer_data, size_t buffer_size) {
  // Rely on the simplest, base flatbuffers verifier. Here is not the place to
  // e.g. use an OpResolver: we just want to make sure the buffer is valid to
  // access the metadata.
  flatbuffers::Verifier verifier = flatbuffers::Verifier(
      reinterpret_cast<const uint8_t*>(buffer_data), buffer_size);
  if (!tflite::VerifyModelBuffer(verifier)) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "The model is not a valid FlatBuffer buffer.",
        MediaPipeTasksStatus::kInvalidFlatBufferError);
  }
  model_ = tflite::GetModel(buffer_data);
  if (model_->metadata() == nullptr) {
    // Not all models have metadata, which is OK. `GetModelMetadata()` then
    // returns nullptr.
    return absl::OkStatus();
  }
  // Look for the "TFLITE_METADATA" field, if any.
  for (int i = 0; i < model_->metadata()->size(); ++i) {
    const auto metadata = model_->metadata()->Get(i);
    if (!metadata->name()) {
      continue;
    }
    if (metadata->name()->str() != kMetadataBufferName) {
      continue;
    }
    const auto buffer_index = metadata->buffer();
    const auto metadata_buffer =
        model_->buffers()->Get(buffer_index)->data()->data();
    if (!tflite::ModelMetadataBufferHasIdentifier(metadata_buffer)) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat(
              "Invalid metadata schema version: expected %s, got %s",
              absl::string_view(tflite::ModelMetadataIdentifier())
                  .substr(
                      0, flatbuffers::FlatBufferBuilder::kFileIdentifierLength),
              // Returned identifier is not null terminated; has to be
              // truncated.
              absl::string_view(
                  flatbuffers::GetBufferIdentifier(metadata_buffer))
                  .substr(
                      0,
                      flatbuffers::FlatBufferBuilder::kFileIdentifierLength)),
          MediaPipeTasksStatus::kMetadataInvalidSchemaVersionError);
    }
    model_metadata_ = tflite::GetModelMetadata(metadata_buffer);
    if (model_metadata_ == nullptr) {
      return CreateStatusWithPayload(StatusCode::kInternal,
                                     "Expected Model Metadata not to be null.");
    }
    return ExtractAssociatedFiles(buffer_data, buffer_size);
    break;
  }
  return absl::OkStatus();
}

absl::Status ModelMetadataExtractor::ExtractAssociatedFiles(
    const char* buffer_data, size_t buffer_size) {
  // Create in-memory read-only zip file.
  ZipReadOnlyMemFile mem_file = ZipReadOnlyMemFile(buffer_data, buffer_size);
  // Open zip.
  unzFile zf = unzOpen2_64(/*path=*/nullptr, &mem_file.GetFileFunc64Def());
  if (zf == nullptr) {
    // It's OK if it fails: this means there are no associated files with this
    // model.
    return absl::OkStatus();
  }
  // Get number of files.
  unz_global_info global_info;
  if (unzGetGlobalInfo(zf, &global_info) != UNZ_OK) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Unable to get zip archive info.",
        MediaPipeTasksStatus::kMetadataAssociatedFileZipError);
  }

  // Browse through files in archive.
  if (global_info.number_entry > 0) {
    int error = unzGoToFirstFile(zf);
    while (error == UNZ_OK) {
      ASSIGN_OR_RETURN(auto zip_file_info, GetCurrentZipFileInfo(zf));
      // Store result in map.
      associated_files_[zip_file_info.name] = absl::string_view(
          buffer_data + zip_file_info.position, zip_file_info.size);
      error = unzGoToNextFile(zf);
    }
    if (error != UNZ_END_OF_LIST_OF_FILE) {
      return CreateStatusWithPayload(
          StatusCode::kUnknown,
          "Unable to read associated file in zip archive.",
          MediaPipeTasksStatus::kMetadataAssociatedFileZipError);
    }
  }
  // Close zip.
  if (unzClose(zf) != UNZ_OK) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Unable to close zip archive.",
        MediaPipeTasksStatus::kMetadataAssociatedFileZipError);
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::string_view> ModelMetadataExtractor::GetAssociatedFile(
    const std::string& filename) const {
  auto it = associated_files_.find(filename);
  if (it == associated_files_.end()) {
    return CreateStatusWithPayload(
        StatusCode::kNotFound,
        absl::StrFormat("No associated file with name: %s", filename),
        MediaPipeTasksStatus::kMetadataAssociatedFileNotFoundError);
  }
  return it->second;
}

absl::StatusOr<std::string> ModelMetadataExtractor::GetModelVersion() const {
  if (model_metadata_ == nullptr) {
    return CreateStatusWithPayload(
        StatusCode::kFailedPrecondition, "No model metadata",
        MediaPipeTasksStatus::kMetadataNotFoundError);
  }
  if (model_metadata_->version() == nullptr) {
    return CreateStatusWithPayload(
        StatusCode::kNotFound, "No version in model metadata",
        MediaPipeTasksStatus::kMetadataNotFoundError);
  }
  return model_metadata_->version()->str();
}

const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorMetadata>>*
ModelMetadataExtractor::GetInputTensorMetadata() const {
  if (model_metadata_ == nullptr ||
      model_metadata_->subgraph_metadata() == nullptr) {
    return nullptr;
  }
  return model_metadata_->subgraph_metadata()
      ->Get(kDefaultSubgraphIndex)
      ->input_tensor_metadata();
}

const tflite::TensorMetadata* ModelMetadataExtractor::GetInputTensorMetadata(
    int index) const {
  return GetItemFromVector<tflite::TensorMetadata>(GetInputTensorMetadata(),
                                                   index);
}

int ModelMetadataExtractor::GetInputTensorCount() const {
  const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorMetadata>>*
      input_tensor_metadata = GetInputTensorMetadata();
  return input_tensor_metadata == nullptr ? 0 : input_tensor_metadata->size();
}

const Vector<Offset<TensorMetadata>>*
ModelMetadataExtractor::GetOutputTensorMetadata() const {
  if (model_metadata_ == nullptr ||
      model_metadata_->subgraph_metadata() == nullptr) {
    return nullptr;
  }
  return model_metadata_->subgraph_metadata()
      ->Get(kDefaultSubgraphIndex)
      ->output_tensor_metadata();
}

const tflite::TensorMetadata* ModelMetadataExtractor::GetOutputTensorMetadata(
    int index) const {
  return GetItemFromVector<tflite::TensorMetadata>(GetOutputTensorMetadata(),
                                                   index);
}

int ModelMetadataExtractor::GetOutputTensorCount() const {
  const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorMetadata>>*
      output_tensor_metadata = GetOutputTensorMetadata();
  return output_tensor_metadata == nullptr ? 0 : output_tensor_metadata->size();
}

const Vector<flatbuffers::Offset<tflite::ProcessUnit>>*
ModelMetadataExtractor::GetInputProcessUnits() const {
  if (model_metadata_ == nullptr ||
      model_metadata_->subgraph_metadata() == nullptr) {
    return nullptr;
  }
  return model_metadata_->subgraph_metadata()
      ->Get(kDefaultSubgraphIndex)
      ->input_process_units();
}

const tflite::ProcessUnit* ModelMetadataExtractor::GetInputProcessUnit(
    int index) const {
  return GetItemFromVector<tflite::ProcessUnit>(GetInputProcessUnits(), index);
}

int ModelMetadataExtractor::GetInputProcessUnitsCount() const {
  const Vector<flatbuffers::Offset<tflite::ProcessUnit>>* input_process_units =
      GetInputProcessUnits();
  return input_process_units == nullptr ? 0 : input_process_units->size();
}

const Vector<flatbuffers::Offset<tflite::ProcessUnit>>*
ModelMetadataExtractor::GetOutputProcessUnits() const {
  if (model_metadata_ == nullptr ||
      model_metadata_->subgraph_metadata() == nullptr) {
    return nullptr;
  }
  return model_metadata_->subgraph_metadata()
      ->Get(kDefaultSubgraphIndex)
      ->output_process_units();
}

const tflite::ProcessUnit* ModelMetadataExtractor::GetOutputProcessUnit(
    int index) const {
  return GetItemFromVector<tflite::ProcessUnit>(GetOutputProcessUnits(), index);
}

int ModelMetadataExtractor::GetOutputProcessUnitsCount() const {
  const Vector<flatbuffers::Offset<tflite::ProcessUnit>>* output_process_units =
      GetOutputProcessUnits();
  return output_process_units == nullptr ? 0 : output_process_units->size();
}

}  // namespace metadata
}  // namespace tasks
}  // namespace mediapipe
