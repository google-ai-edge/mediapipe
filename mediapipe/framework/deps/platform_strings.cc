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

#include "mediapipe/framework/deps/platform_strings.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <errno.h>
#include <string.h>
#endif  // _WIN32

#include <string>

namespace mediapipe {
#ifdef _WIN32
std::string FormatLastError() {
  DWORD message_id = GetLastError();
  if (message_id == 0) {
    return std::string("(no error reported)");
  }

  LPSTR message_buffer = nullptr;
  DWORD size = FormatMessage(
      /*dwFlags=*/(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                   FORMAT_MESSAGE_IGNORE_INSERTS),
      /*lpSource=*/NULL,
      /*dwMessageId=*/message_id,
      /*dwLanguageId=*/MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      /*lpBuffer=*/(LPSTR)&message_buffer,
      /*nSize=*/0,
      /*Arguments=*/NULL);
  if (size == 0) {
    return "(error while trying to format the error message)";
  }

  std::string message(message_buffer, size);
  LocalFree(message_buffer);
  return NativeToUtf8(message);
}
#else
std::string FormatLastError() { return strerror(errno); }
#endif  // _WIN32
}  // namespace mediapipe
