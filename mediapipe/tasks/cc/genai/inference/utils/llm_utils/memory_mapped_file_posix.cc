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
#include <sys/mman.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/memory_mapped_file.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/scoped_file.h"

namespace mediapipe::tasks::genai::llm_utils {
namespace {

class MemoryMappedFilePosix : public MemoryMappedFile {
 public:
  MemoryMappedFilePosix(uint64_t length, void* data)
      : length_(length), data_(data) {}
  ~MemoryMappedFilePosix() override { munmap(data_, length_); }

  uint64_t length() override { return length_; }

  void* data() override { return data_; }

 private:
  uint64_t length_;
  void* data_;
};

}  // namespace

// static
size_t MemoryMappedFile::GetOffsetAlignment() { return getpagesize(); }

// static
absl::StatusOr<std::unique_ptr<MemoryMappedFile>> MemoryMappedFile::Create(
    absl::string_view path) {
  MP_ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::Open(path));
  return Create(scoped_file.file());
}

// static
absl::StatusOr<std::unique_ptr<MemoryMappedFile>> MemoryMappedFile::Create(
    int file, uint64_t offset, uint64_t length, absl::string_view key) {
  RET_CHECK_EQ(offset % GetOffsetAlignment(), 0)
      << "Offset must be a multiple of page size : " << offset << ", "
      << GetOffsetAlignment();

  size_t file_size = lseek(file, 0, SEEK_END);
  RET_CHECK_GE(file_size, length + offset) << "Length and offset too large.";
  if (length == 0) {
    length = file_size - offset;
  }

  // Some Mac versions (Macbook Pro 2019) have very bad performance with
  // MAP_PRIVATE, so use MAP_SHARED here. The Metal API for importing host
  // memory doesn't require it to be writable, so it's fine to just use
  // PROT_READ here.
#if defined(__APPLE__)
  void* data = mmap(nullptr, length, PROT_READ, MAP_SHARED, file, offset);
#else
  void* data =
      mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_PRIVATE, file, offset);
#endif
  RET_CHECK_NE(data, MAP_FAILED) << "Failed to map, error: " << strerror(errno);
  RET_CHECK_NE(data, nullptr) << "Failed to map.";
  RET_CHECK_EQ(madvise(data, length, MADV_WILLNEED), 0) << "madvise failed.";

  return std::make_unique<MemoryMappedFilePosix>(length, data);
}

absl::StatusOr<std::unique_ptr<MemoryMappedFile>>
MemoryMappedFile::CreateMutable(absl::string_view path) {
  MP_ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::OpenWritable(path));
  int fd = scoped_file.file();
  RET_CHECK_GE(fd, 0) << "open() failed: " << path;
  auto close_fd = absl::MakeCleanup([fd] { close(fd); });

  size_t length = lseek(fd, 0, SEEK_END);

  void* data = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  RET_CHECK_NE(data, MAP_FAILED)
      << "Failed to map " << path << ", error: " << strerror(errno);
  RET_CHECK_NE(data, nullptr) << "Failed to map: " << path;

  return std::make_unique<MemoryMappedFilePosix>(length, data);
}

}  // namespace mediapipe::tasks::genai::llm_utils
