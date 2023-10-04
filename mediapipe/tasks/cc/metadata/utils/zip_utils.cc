/* Copyright 2022 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/metadata/utils/zip_utils.h"

#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "contrib/minizip/ioapi.h"
#include "contrib/minizip/unzip.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/metadata/utils/zip_readonly_mem_file.h"

namespace mediapipe {
namespace tasks {
namespace metadata {

namespace {

using ::absl::StatusCode;

// Wrapper function around calls to unzip to avoid repeating conversion logic
// from error code to Status.
absl::Status UnzipErrorToStatus(int error) {
  if (error != UNZ_OK) {
    return CreateStatusWithPayload(StatusCode::kUnknown,
                                   "Unable to read the file in zip archive.",
                                   MediaPipeTasksStatus::kFileZipError);
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
  absl::Cleanup unzipper_closer = [zf]() {
    auto status = UnzipErrorToStatus(unzCloseCurrentFile(zf));
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to close the current zip file: " << status;
    }
  };
  if (method != Z_NO_COMPRESSION) {
    return CreateStatusWithPayload(StatusCode::kUnknown,
                                   "Expected uncompressed zip archive.",
                                   MediaPipeTasksStatus::kFileZipError);
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
    return CreateStatusWithPayload(StatusCode::kUnknown,
                                   "Unable to read file in zip archive.",
                                   MediaPipeTasksStatus::kFileZipError);
  }

  // Perform the cleanup manually for error propagation.
  std::move(unzipper_closer).Cancel();
  // Close file and return.
  MP_RETURN_IF_ERROR(UnzipErrorToStatus(unzCloseCurrentFile(zf)));

  ZipFileInfo result{};
  result.name = file_name;
  result.position = position;
  result.size = file_info.uncompressed_size;
  return result;
}

}  // namespace

absl::Status ExtractFilesfromZipFile(
    const char* buffer_data, const size_t buffer_size,
    absl::flat_hash_map<std::string, absl::string_view>* files) {
  // Create in-memory read-only zip file.
  ZipReadOnlyMemFile mem_file = ZipReadOnlyMemFile(buffer_data, buffer_size);
  // Open zip.
  unzFile zf = unzOpen2_64(/*path=*/nullptr, &mem_file.GetFileFunc64Def());
  if (zf == nullptr) {
    return CreateStatusWithPayload(StatusCode::kUnknown,
                                   "Unable to open zip archive.",
                                   MediaPipeTasksStatus::kFileZipError);
  }
  absl::Cleanup unzipper_closer = [zf]() {
    if (unzClose(zf) != UNZ_OK) {
      ABSL_LOG(ERROR) << "Unable to close zip archive.";
    }
  };
  // Get number of files.
  unz_global_info global_info;
  if (unzGetGlobalInfo(zf, &global_info) != UNZ_OK) {
    return CreateStatusWithPayload(StatusCode::kUnknown,
                                   "Unable to get zip archive info.",
                                   MediaPipeTasksStatus::kFileZipError);
  }

  // Browse through files in archive.
  if (global_info.number_entry > 0) {
    int error = unzGoToFirstFile(zf);
    while (error == UNZ_OK) {
      MP_ASSIGN_OR_RETURN(auto zip_file_info, GetCurrentZipFileInfo(zf));
      // Store result in map.
      (*files)[zip_file_info.name] = absl::string_view(
          buffer_data + zip_file_info.position, zip_file_info.size);
      error = unzGoToNextFile(zf);
    }
    if (error != UNZ_END_OF_LIST_OF_FILE) {
      return CreateStatusWithPayload(
          StatusCode::kUnknown,
          "Unable to read associated file in zip archive.",
          MediaPipeTasksStatus::kFileZipError);
    }
  }
  // Perform the cleanup manually for error propagation.
  std::move(unzipper_closer).Cancel();
  // Close zip.
  if (unzClose(zf) != UNZ_OK) {
    return CreateStatusWithPayload(StatusCode::kUnknown,
                                   "Unable to close zip archive.",
                                   MediaPipeTasksStatus::kFileZipError);
  }
  return absl::OkStatus();
}

void SetExternalFile(const absl::string_view& file_content,
                     core::proto::ExternalFile* model_file, bool is_copy) {
  if (is_copy) {
    std::string str_content{file_content};
    model_file->set_file_content(str_content);
  } else {
    auto pointer = reinterpret_cast<uint64_t>(file_content.data());
    model_file->mutable_file_pointer_meta()->set_pointer(pointer);
    model_file->mutable_file_pointer_meta()->set_length(file_content.length());
  }
}

}  // namespace metadata
}  // namespace tasks
}  // namespace mediapipe
