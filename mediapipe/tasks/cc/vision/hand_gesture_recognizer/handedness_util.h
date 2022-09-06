/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_GESTURE_RECOGNIZER_HADNDEDNESS_UTILS_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_GESTURE_RECOGNIZER_HADNDEDNESS_UTILS_H_

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/classification.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {

bool IsLeftHand(const mediapipe::Classification& c);

bool IsRightHand(const mediapipe::Classification& c);

absl::StatusOr<float> GetLeftHandScore(
    const mediapipe::ClassificationList& classification_list);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_GESTURE_RECOGNIZER_HADNDEDNESS_UTILS_H_
