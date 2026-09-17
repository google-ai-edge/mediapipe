/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/retrieval/semantic_retriever/image_decoder.h"

#include <fstream>
#include <ios>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace mediapipe::tasks::retrieval {

absl::StatusOr<std::string> ImageDecoder::DecodeImageBytes(
    absl::string_view path) {
  std::ifstream file(std::string(path), std::ios::binary | std::ios::ate);
  if (!file) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open image file: ", path));
  }
  std::streamsize size = file.tellg();
  if (size < 0) {
    return absl::InternalError(
        absl::StrCat("Failed to determine size of image file: ", path));
  }
  if (size == 0) {
    return "";
  }
  file.seekg(0, std::ios::beg);
  std::string buffer(size, '\0');
  if (!file.read(&buffer[0], size)) {
    return absl::InternalError(
        absl::StrCat("Failed to read image file: ", path));
  }
  return buffer;
}

}  // namespace mediapipe::tasks::retrieval
