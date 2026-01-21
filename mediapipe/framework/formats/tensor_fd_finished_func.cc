#include "mediapipe/framework/formats/tensor_fd_finished_func.h"

#include "absl/log/absl_log.h"
#include "absl/time/time.h"
#include "mediapipe/framework/formats/shared_fd.h"
#include "mediapipe/util/sync_wait.h"

namespace mediapipe {
namespace {

inline bool IsFinished(bool wait, const SharedFd& fd) {
  const auto is_signaled = IsSignaled(fd);
  if (!is_signaled.ok()) {
    ABSL_LOG(ERROR) << is_signaled.status();
    return false;
  }
  if (!wait) {
    return *is_signaled;
  }

  if (*is_signaled) {
    return true;
  }

  auto status = SyncWait(fd, absl::InfiniteDuration());
  if (!status.ok()) {
    ABSL_LOG(ERROR) << status;
    return false;
  }
  return true;
}

}  // namespace

bool FdFinishedFunc::operator()(bool wait) { return IsFinished(wait, fd_); }

bool MultipleFdsFinishedFunc::operator()(bool wait) {
  for (const auto& fd : fds_) {
    if (!IsFinished(wait, fd)) {
      return false;
    }
  }
  return true;
}

}  // namespace mediapipe
