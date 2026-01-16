#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_FD_FINISHED_FUNC_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_FD_FINISHED_FUNC_H_

#include <utility>
#include <vector>

#include "mediapipe/framework/formats/shared_fd.h"

namespace mediapipe {
// Funcs below to be used by Tensor::SetWritingFinishedFD and
// Tensor::SetReadingFinishedFunc.
class FdFinishedFunc {
 public:
  explicit FdFinishedFunc(SharedFd fd) : fd_(std::move(fd)) {}

  bool operator()(bool wait);

 private:
  // SharedFd to support copy construction required by std::function.
  SharedFd fd_;
};

class MultipleFdsFinishedFunc {
 public:
  explicit MultipleFdsFinishedFunc(std::vector<SharedFd> fds)
      : fds_(std::move(fds)) {}

  bool operator()(bool wait);

 private:
  std::vector<SharedFd> fds_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_FD_FINISHED_FUNC_H_
