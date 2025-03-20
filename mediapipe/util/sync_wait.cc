#include "mediapipe/util/sync_wait.h"

#include <sys/poll.h>

#include <cerrno>
#include <cstdint>
#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "mediapipe/framework/formats/shared_fd.h"
#include "mediapipe/framework/formats/unique_fd.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {

absl::Status SyncWait(int fd, absl::Duration timeout) {
  RET_CHECK_GE(fd, 0) << "Invalid file descriptor.";

  int64_t timeout_millis = -1;  // Infinite for poll.
  if (timeout != absl::InfiniteDuration()) {
    timeout_millis = absl::ToInt64Milliseconds(timeout);

    constexpr int kIntMax = std::numeric_limits<int>::max();
    RET_CHECK_LE(timeout_millis, kIntMax)
        << "Timeout cannot be greater than: " << kIntMax;
  }

  struct pollfd fds;
  fds.fd = fd;
  fds.events = POLLIN;
  int ret;
  do {
    ret = poll(&fds, 1, timeout_millis);
    if (ret == 1) {
      RET_CHECK((fds.revents & POLLERR) == 0);
      RET_CHECK((fds.revents & POLLNVAL) == 0);
      return absl::OkStatus();
    } else if (ret == 0) {
      return absl::DeadlineExceededError(
          absl::StrFormat("Timeout expired: %d.", timeout_millis));
    }
  } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

  return absl::ErrnoToStatus(errno,
                             absl::StrFormat("Failed to wait for fd: %d.", fd));
}

absl::Status SyncWait(const UniqueFd& fd, absl::Duration timeout) {
  RET_CHECK(fd.IsValid());
  return SyncWait(fd.Get(), timeout);
}

absl::Status SyncWait(const SharedFd& fd, absl::Duration timeout) {
  RET_CHECK(fd);
  return SyncWait(fd.Get(), timeout);
}

absl::StatusOr<bool> IsSignaled(int fd) {
  RET_CHECK_GE(fd, 0) << "Invalid file descriptor.";

  struct pollfd fds;
  fds.fd = fd;
  fds.events = POLLIN;
  int ret;
  do {
    ret = poll(&fds, 1, /*timeout_millis=*/0);
    if (ret == 1) {
      RET_CHECK((fds.revents & POLLERR) == 0);
      RET_CHECK((fds.revents & POLLNVAL) == 0);
      return true;
    } else if (ret == 0) {
      return false;
    }
  } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

  return absl::ErrnoToStatus(
      errno, absl::StrFormat("Failed to check if fd: %d is signaled.", fd));
}

absl::StatusOr<bool> IsSignaled(const UniqueFd& fd) {
  return IsSignaled(fd.Get());
}

absl::StatusOr<bool> IsSignaled(const SharedFd& fd) {
  RET_CHECK(fd);
  return IsSignaled(fd.Get());
}

}  // namespace mediapipe
