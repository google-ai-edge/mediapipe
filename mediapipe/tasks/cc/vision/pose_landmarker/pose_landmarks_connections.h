/* Copyright 2023 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_LANDMARKS_CONNECTIONS_H_
#define MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_LANDMARKS_CONNECTIONS_H_

#include <array>

namespace mediapipe {
namespace tasks {
namespace vision {
namespace pose_landmarker {

static constexpr std::array<std::array<int, 2>, 34> kPoseLandmarksConnections{{
    {1, 2},   {0, 1},   {2, 3},   {3, 7},   {0, 4},   {4, 5},   {5, 6},
    {6, 8},   {9, 10},  {11, 12}, {11, 13}, {13, 15}, {15, 17}, {15, 19},
    {15, 21}, {17, 19}, {12, 14}, {14, 16}, {16, 18}, {16, 20}, {16, 22},
    {18, 20}, {11, 23}, {12, 24}, {23, 24}, {23, 25}, {24, 26}, {25, 27},
    {26, 28}, {27, 29}, {28, 30}, {29, 31}, {30, 32}, {27, 31},
}};

}  // namespace pose_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_LANDMARKS_CONNECTIONS_H_
