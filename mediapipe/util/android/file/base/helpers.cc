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

#include "mediapipe/util/android/file/base/helpers.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>

#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace file {

namespace {

// Simple wrapper class that automatically calls close() on a file descriptor
// when the wrapper goes out of scope.
class FdCloser {
 public:
  explicit FdCloser(int fd) : fd_(fd) {}
  ~FdCloser() { close(fd_); }

 private:
  int fd_;
};

}  // namespace

// Read contents of a file to a string.
absl::Status GetContents(int fd, std::string* output) {
  // Determine the length of the file.
  struct stat buf;
  if (fstat(fd, &buf) != 0) {
    return absl::Status(absl::StatusCode::kUnknown,
                        "Failed to get file status");
  }
  if (buf.st_size < 0 || buf.st_size > SIZE_MAX) {
    return absl::Status(absl::StatusCode::kInternal, "Invalid file size");
  }
  size_t length = buf.st_size;

  // Load the data.
  output->resize(length);
  char* output_ptr = &output->front();
  while (length != 0) {
    const ssize_t nread = read(fd, output_ptr, length);
    if (nread <= 0) {
      return absl::Status(absl::StatusCode::kUnknown, "Failed to read file");
    }
    output_ptr += nread;
    length -= nread;
  }
  return absl::OkStatus();
}

// Read contents of a file to a string.
absl::Status GetContents(absl::string_view file_name, std::string* output,
                         const file::Options& /*options*/) {
  int fd = open(std::string(file_name).c_str(), O_RDONLY);
  if (fd < 0) {
    return absl::Status(absl::StatusCode::kUnknown,
                        "Failed to open file: " + std::string(file_name));
  }

  FdCloser closer(fd);
  return GetContents(fd, output);
}

absl::Status GetContents(absl::string_view file_name, std::string* output) {
  return GetContents(file_name, output, file::Defaults());
}

absl::Status SetContents(absl::string_view file_name, absl::string_view content,
                         const file::Options& options) {
  // Options are currently ignored.
  return SetContents(file_name, content);
}

}  // namespace file
}  // namespace mediapipe
