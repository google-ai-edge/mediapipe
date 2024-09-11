#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_ANDROID_UNIQUE_FD_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_ANDROID_UNIQUE_FD_H_

#include <unistd.h>

#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"

#if (__ANDROID_API__ >= 29) && defined(__BIONIC__) && !defined(NDEBUG)
#define MEDIAPIPE_USE_FDSAN 1
#include <android/fdsan.h>
#endif  // (__ANDROID_API__ >= 29) && defined(__BIONIC__) && !defined(NDEBUG)

namespace mediapipe {

// Implementation of a unique file descriptor wrapper inspired by
// https://android.googlesource.com/platform/bionic/+/master/docs/fdsan.md
//
// This class is a wrapper around a file descriptor that ensures that the
// descriptor is closed when the wrapper goes out of scope.
//
// This class is not thread-safe.
class UniqueFd {
 public:
  UniqueFd() = default;

  // Wraps and takes ownership of the given file descriptor.
  explicit UniqueFd(int fd) { Reset(fd); }

  UniqueFd(const UniqueFd& copy) = delete;
  UniqueFd(UniqueFd&& move) { *this = std::move(move); }

  // Closes the wrapped file descriptor during destruction if it is valid.
  ~UniqueFd() { Reset(); }

  UniqueFd& operator=(const UniqueFd& copy) = delete;
  UniqueFd& operator=(UniqueFd&& move) {
    if (this == &move) {
      return *this;
    }

    Reset();

    if (move.fd_ != -1) {
      fd_ = move.fd_;
      move.fd_ = -1;
#if defined(MEDIAPIPE_USE_FDSAN)
      // Acquire ownership from the moved-from object.
      FdsanExchangeTag(fd_, move.Tag(), Tag());
#endif  // MEDIAPIPE_USE_FDSAN
    }

    return *this;
  }

  // Returns a non-owned file descriptor.
  int Get() { return fd_; }

  // Checks if a valid file descriptor is wrapped.
  bool IsValid() const { return fd_ >= 0; }

  // Releases ownership of the file descriptor and returns it.
  ABSL_MUST_USE_RESULT int Release() {
    if (!IsValid()) {
      return -1;
    }

    int fd = fd_;
    fd_ = -1;
#if defined(MEDIAPIPE_USE_FDSAN)
    // Release ownership.
    FdsanExchangeTag(fd, Tag(), 0);
#endif  // MEDIAPIPE_USE_FDSAN
    return fd;
  }

  // Closes a wrapped file descriptor and resets the wrapper.
  void Reset(int new_fd = -1) {
    if (IsValid()) {
#if defined(MEDIAPIPE_USE_FDSAN)
      FdsanClose(fd_, Tag());
#else
      if (::close(fd_) != 0) {
        ABSL_LOG(ERROR) << "Failed to close fd: " << fd_;
      }
#endif  // MEDIAPIPE_USE_FDSAN
      fd_ = -1;
    }

    if (new_fd != -1) {
      fd_ = new_fd;
#if defined(MEDIAPIPE_USE_FDSAN)
      // Acquire ownership of the presumably unowned fd.
      FdsanExchangeTag(fd_, 0, Tag());
#endif  // MEDIAPIPE_USE_FDSAN
    }
  }

 private:
  int fd_ = -1;

#if defined(MEDIAPIPE_USE_FDSAN)
  // Address of the object is used as tag.
  uint64_t Tag() { return reinterpret_cast<uint64_t>(this); }

  // These functions are marked with __attribute__((weak)), so that their
  // availability can be determined at runtime. These wrappers will use them
  // if available, and fall back to no-ops or regular close on devices older
  // than API level 29 or non-bionic or non-production builds.
  static void FdsanExchangeTag(int fd, uint64_t old_tag, uint64_t new_tag) {
    if (android_fdsan_exchange_owner_tag) {
      android_fdsan_exchange_owner_tag(fd, old_tag, new_tag);
    }
  }

  static void FdsanClose(int fd, uint64_t tag) {
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
#endif  // MEDIAPIPE_USE_FDSAN
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_ANDROID_UNIQUE_FD_H_
