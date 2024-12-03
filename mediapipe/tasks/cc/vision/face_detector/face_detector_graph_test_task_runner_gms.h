#ifndef MEDIAPIPE_TASKS_CC_VISION_FACE_DETECTOR_FACE_DETECTOR_GRAPH_TEST_TASK_RUNNER_GMS_H_
#define MEDIAPIPE_TASKS_CC_VISION_FACE_DETECTOR_FACE_DETECTOR_GRAPH_TEST_TASK_RUNNER_GMS_H_

#include <memory>

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
                    std::unique_ptr<tflite::OpResolver> op_resolver = nullptr);

}  // namespace test_util
}  // namespace face_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_FACE_DETECTOR_FACE_DETECTOR_GRAPH_TEST_TASK_RUNNER_GMS_H_
