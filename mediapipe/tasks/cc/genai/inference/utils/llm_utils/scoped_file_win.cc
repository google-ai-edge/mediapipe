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

#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/scoped_file.h"

namespace mediapipe::tasks::genai::llm_utils {
namespace {
std::wstring Utf8ToWideChar(absl::string_view utf8str) {
  int size_required = MultiByteToWideChar(CP_UTF8, 0, utf8str.data(),
                                          (int)utf8str.size(), nullptr, 0);
  std::wstring ws_translated_str(size_required, 0);
  MultiByteToWideChar(CP_UTF8, 0, utf8str.data(), (int)utf8str.size(),
                      &ws_translated_str[0], size_required);
  return ws_translated_str;
}

absl::StatusOr<ScopedFile> OpenImpl(absl::string_view path,
                                    DWORD file_attribute_flag) {
  std::wstring ws_path = Utf8ToWideChar(path);
  DWORD file_flags =
      file_attribute_flag | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN;

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  DWORD access = GENERIC_READ;
  if (file_attribute_flag == FILE_ATTRIBUTE_NORMAL) {
    access |= GENERIC_WRITE;
  }
  HANDLE hfile = ::CreateFileW(ws_path.c_str(), access, share_mode, nullptr,
                               OPEN_EXISTING, file_flags, nullptr);
  RET_CHECK_NE(hfile, INVALID_HANDLE_VALUE) << "Failed to open: " << path;
  return ScopedFile(hfile);
}
}  // namespace

const HANDLE ScopedFile::kInvalidPlatformFile = INVALID_HANDLE_VALUE;

// static
absl::StatusOr<ScopedFile> ScopedFile::Open(absl::string_view path) {
  return OpenImpl(path, FILE_ATTRIBUTE_READONLY);
}

// static
absl::StatusOr<ScopedFile> ScopedFile::OpenWritable(absl::string_view path) {
  return OpenImpl(path, FILE_ATTRIBUTE_NORMAL);
}

// static
void ScopedFile::CloseFile(HANDLE file) { ::CloseHandle(file); }

}  // namespace mediapipe::tasks::genai::llm_utils
