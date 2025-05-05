#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_FD_FINISHED_FUNC_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_FD_FINISHED_FUNC_H_

#include <utility>

#include "absl/log/absl_log.h"
#include "absl/time/time.h"
#include "mediapipe/framework/formats/shared_fd.h"
#include "mediapipe/framework/formats/unique_fd.h"
#include "mediapipe/util/sync_wait.h"

namespace mediapipe {
// Intended to be used by Tensor::SetWritingFinishedFD and
// Tensor::SetReadingFinishedFunc.
class FdFinishedFunc {
 public:
  explicit FdFinishedFunc(UniqueFd fd) : fd_(std::move(fd)) {}

  bool operator()(bool wait) {
    const auto is_signaled = IsSignaled(fd_);
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

    auto status = SyncWait(fd_, absl::InfiniteDuration());
    if (!status.ok()) {
      ABSL_LOG(ERROR) << status;
      return false;
    }
    return true;
  }

 private:
  // SharedFd to support copy construction required by std::function.
  SharedFd fd_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_FD_FINISHED_FUNC_H_
