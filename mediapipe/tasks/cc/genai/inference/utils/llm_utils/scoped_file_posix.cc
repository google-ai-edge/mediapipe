// Copyright 2024 The MediaPipe Authors.
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

#include <fcntl.h>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/scoped_file.h"

namespace mediapipe::tasks::genai::llm_utils {

// static
absl::StatusOr<ScopedFile> ScopedFile::Open(absl::string_view path) {
  int fd = open(path.data(), O_RDONLY);
  RET_CHECK_GE(fd, 0) << "open() failed: " << path;
  return ScopedFile(fd);
}

// static
absl::StatusOr<ScopedFile> ScopedFile::OpenWritable(absl::string_view path) {
  int fd = open(path.data(), O_RDWR);
  RET_CHECK_GE(fd, 0) << "open() failed: " << path;
  return ScopedFile(fd);
}

// static
void ScopedFile::CloseFile(int file) { close(file); }

}  // namespace mediapipe::tasks::genai::llm_utils
