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

#include <windows.h>

#include "absl/cleanup/cleanup.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/memory_mapped_file.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/scoped_file.h"

namespace mediapipe::tasks::genai::llm_utils {
namespace {

class MemoryMappedFileWin : public MemoryMappedFile {
 public:
  MemoryMappedFileWin(HANDLE hmap, uint64_t length, void* data)
      : hmap_(hmap), length_(length), data_(data) {}
  ~MemoryMappedFileWin() override {
    ::UnmapViewOfFile(data_);
    ::CloseHandle(hmap_);
  }

  uint64_t length() override { return length_; }

  void* data() override { return data_; }

 private:
  HANDLE hmap_;
  uint64_t length_;
  void* data_;
};

absl::StatusOr<std::unique_ptr<MemoryMappedFile>> CreateImpl(HANDLE hfile,
                                                             uint64_t offset,
                                                             uint64_t length,
                                                             const char* key,
                                                             bool writable) {
  RET_CHECK_EQ(offset % MemoryMappedFile::GetOffsetAlignment(), 0)
      << "Offset must be a multiple of allocation granularity: " << offset
      << ", " << MemoryMappedFile::GetOffsetAlignment();

  LARGE_INTEGER size;
  RET_CHECK(::GetFileSizeEx(hfile, &size)) << "Failed to get size.";
  int64_t file_size = static_cast<int64_t>(size.QuadPart);
  RET_CHECK_GE(file_size, length + offset) << "Length and offset too large.";
  if (length == 0) {
    length = file_size - offset;
  }

  DWORD access = FILE_MAP_COPY;
  DWORD protect = PAGE_WRITECOPY;
  if (writable) {
    access = FILE_MAP_ALL_ACCESS;
    protect = PAGE_READWRITE;
  }

  HANDLE hmap = ::OpenFileMappingA(access, false, key);
  if (hmap == NULL) {
    hmap = ::CreateFileMappingA(hfile, nullptr, protect, 0, 0, key);
  }

  RET_CHECK(hmap) << "Failed to create mapping.";
  auto close_hmap = absl::MakeCleanup([hmap] { ::CloseHandle(hmap); });

  ULARGE_INTEGER map_start = {};
  map_start.QuadPart = offset;
  void* mapped_region = ::MapViewOfFile(hmap, access, map_start.HighPart,
                                        map_start.LowPart, length);
  RET_CHECK(mapped_region) << "Failed to map.";

  std::move(close_hmap).Cancel();

  return std::make_unique<MemoryMappedFileWin>(hmap, length, mapped_region);
}

}  // namespace

// static
size_t MemoryMappedFile::GetOffsetAlignment() {
  SYSTEM_INFO sys_info;
  ::GetSystemInfo(&sys_info);
  return sys_info.dwAllocationGranularity;
}

// static
absl::StatusOr<std::unique_ptr<MemoryMappedFile>> MemoryMappedFile::Create(
    absl::string_view path) {
  MP_ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::Open(path));
  return CreateImpl(scoped_file.file(), 0, 0, nullptr, /*writable=*/false);
}

// static
absl::StatusOr<std::unique_ptr<MemoryMappedFile>> MemoryMappedFile::Create(
    HANDLE file, uint64_t offset, uint64_t length, absl::string_view key) {
  return CreateImpl(file, offset, length, key.empty() ? nullptr : key.data(),
                    /*writable=*/false);
}

// static
absl::StatusOr<std::unique_ptr<MemoryMappedFile>>
MemoryMappedFile::CreateMutable(absl::string_view path) {
  MP_ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::OpenWritable(path));
  return CreateImpl(scoped_file.file(), 0, 0, nullptr, /*writable=*/true);
}

// static
absl::StatusOr<std::unique_ptr<MemoryMappedFile>>
MemoryMappedFile::CreateMutable(HANDLE file, uint64_t offset, uint64_t length,
                                absl::string_view key) {
  return CreateImpl(file, offset, length, key.empty() ? nullptr : key.data(),
                    /*writable=*/true);
}

}  // namespace mediapipe::tasks::genai::llm_utils
