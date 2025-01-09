#include "mediapipe/tasks/cc/vision/face_detector/face_detector_graph_test_task_runner_gms.h"

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_detector {
namespace test_util {

absl::StatusOr<std::unique_ptr<mediapipe::tasks::core::TaskRunner>>
CreateTaskRunnerGms(mediapipe::CalculatorGraphConfig config,
                    std::unique_ptr<tflite::OpResolver> op_resolver) {
  return mediapipe::tasks::core::TaskRunner::Create(std::move(config),
                                                    std::move(op_resolver));
}

}  // namespace test_util
}  // namespace face_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
