#ifndef MEDIAPIPE_TASKS_CC_COMPONENTS_CONTAINERS_KEYPOINT_H_
#define MEDIAPIPE_TASKS_CC_COMPONENTS_CONTAINERS_KEYPOINT_H_

#include <optional>
#include <string>

namespace mediapipe::tasks::components::containers {

// A keypoint, defined by the coordinates (x, y), normalized
// by the image dimensions.
struct NormalizedKeypoint {
  // x in normalized image coordinates.
  float x;
  // y in normalized image coordinates.
  float y;
  // optional label of the keypoint.
  std::optional<std::string> label;
  // optional score of the keypoint.
  std::optional<float> score;
};

}  // namespace mediapipe::tasks::components::containers

#endif  // MEDIAPIPE_TASKS_CC_COMPONENTS_CONTAINERS_KEYPOINT_H_
