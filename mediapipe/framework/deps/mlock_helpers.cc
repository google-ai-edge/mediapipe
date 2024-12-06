#include "mediapipe/framework/deps/mlock_helpers.h"

#include <cstddef>

#ifdef _WIN32
// clang-format off
#include <windows.h>  // Must come before other Windows headers.
// clang-format on
#include <memoryapi.h>
#else
#include <sys/mman.h>
#endif

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/deps/platform_strings.h"

namespace mediapipe {
#ifdef _WIN32
absl::Status LockMemory(const void* base_address, size_t length) {
  BOOL status = VirtualLock(const_cast<LPVOID>(base_address), length);
  if (!status) {
    return absl::UnavailableError(
        absl::StrCat("Failed to lock pages in memory: ", FormatLastError()));
  }
  return absl::OkStatus();
}

absl::Status UnlockMemory(const void* base_address, size_t length) {
  BOOL status = VirtualUnlock(const_cast<LPVOID>(base_address), length);
  if (!status) {
    return absl::UnavailableError(
        absl::StrCat("Failed to unlock memory pages: ", FormatLastError()));
  }
  return absl::OkStatus();
}
#else   // _WIN32
absl::Status LockMemory(const void* base_address, size_t length) {
  int status = mlock(base_address, length);
  if (status < 0) {
    return absl::UnavailableError(
        absl::StrCat("Failed to lock pages in memory: ", FormatLastError()));
  }
  return absl::OkStatus();
}

absl::Status UnlockMemory(const void* base_address, size_t length) {
  int status = munlock(base_address, length);
  if (status < 0) {
    return absl::UnavailableError(
        absl::StrCat("Failed to unlock memory pages: ", FormatLastError()));
  }
  return absl::OkStatus();
}
#endif  // _WIN32
}  // namespace mediapipe
