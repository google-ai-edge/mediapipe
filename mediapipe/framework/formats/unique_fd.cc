#include "mediapipe/framework/formats/unique_fd.h"

#include <unistd.h>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/port/ret_check.h"

#if (__ANDROID_API__ >= 29) && defined(__BIONIC__) && !defined(NDEBUG)
#define MEDIAPIPE_UNIQUE_FD_USE_FDSAN 1

#include <android/fdsan.h>

#include <cstdint>

#endif  // (__ANDROID_API__ >= 29) && defined(__BIONIC__) && !defined(NDEBUG)

namespace mediapipe {

namespace {

#if defined(MEDIAPIPE_UNIQUE_FD_USE_FDSAN)
// Address of the object is used as tag.
uint64_t Tag(UniqueFd* fd) { return reinterpret_cast<uint64_t>(fd); }

// These functions are marked with __attribute__((weak)), so that their
// availability can be determined at runtime. These wrappers will use them
// if available, and fall back to no-ops or regular close on devices older
// than API level 29 or non-bionic or non-production builds.
void FdsanExchangeTag(int fd, uint64_t old_tag, uint64_t new_tag) {
  if (android_fdsan_exchange_owner_tag) {
    android_fdsan_exchange_owner_tag(fd, old_tag, new_tag);
  }
}

void FdsanClose(int fd, uint64_t tag) {
  if (android_fdsan_close_with_tag) {
    if (android_fdsan_close_with_tag(fd, tag) != 0) {
      ABSL_LOG(ERROR) << "Failed to close fd: " << fd;
    }
    return;
  }
  if (::close(fd) != 0) {
    ABSL_LOG(ERROR) << "Failed to close fd: " << fd;
  }
}
#endif  // MEDIAPIPE_UNIQUE_FD_USE_FDSAN

}  // namespace

UniqueFd& UniqueFd::operator=(UniqueFd&& move) {
  if (this == &move) {
    return *this;
  }

  Reset();

  if (move.fd_ != -1) {
    fd_ = move.fd_;
    move.fd_ = -1;
#if defined(MEDIAPIPE_UNIQUE_FD_USE_FDSAN)
    // Acquire ownership from the moved-from object.
    FdsanExchangeTag(fd_, Tag(&move), Tag(this));
#endif  // MEDIAPIPE_UNIQUE_FD_USE_FDSAN
  }

  return *this;
}

absl::StatusOr<UniqueFd> UniqueFd::Dup() const {
  RET_CHECK(IsValid());
  int dup_fd = dup(Get());
  return UniqueFd(dup_fd);
}

// Releases ownership of the file descriptor and returns it.
ABSL_MUST_USE_RESULT int UniqueFd::Release() {
  if (!IsValid()) {
    return -1;
  }

  int fd = fd_;
  fd_ = -1;
#if defined(MEDIAPIPE_UNIQUE_FD_USE_FDSAN)
  // Release ownership.
  FdsanExchangeTag(fd, Tag(this), 0);
#endif  // MEDIAPIPE_UNIQUE_FD_USE_FDSAN
  return fd;
}

// Closes a wrapped file descriptor and resets the wrapper.
void UniqueFd::Reset(int new_fd) {
  if (IsValid()) {
#if defined(MEDIAPIPE_UNIQUE_FD_USE_FDSAN)
    FdsanClose(fd_, Tag(this));
#else
    if (::close(fd_) != 0) {
      ABSL_LOG(ERROR) << "Failed to close fd: " << fd_;
    }
#endif  // MEDIAPIPE_UNIQUE_FD_USE_FDSAN
    fd_ = -1;
  }

  if (new_fd != -1) {
    fd_ = new_fd;
#if defined(MEDIAPIPE_UNIQUE_FD_USE_FDSAN)
    // Acquire ownership of the presumably unowned fd.
    FdsanExchangeTag(fd_, 0, Tag(this));
#endif  // MEDIAPIPE_UNIQUE_FD_USE_FDSAN
  }
}

}  // namespace mediapipe

#ifdef MEDIAPIPE_UNIQUE_FD_USE_FDSAN
#undef MEDIAPIPE_UNIQUE_FD_USE_FDSAN
#endif
