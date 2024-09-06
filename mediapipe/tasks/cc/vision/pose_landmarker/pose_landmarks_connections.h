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

inline constexpr std::array<std::array<int, 2>, 35> kPoseLandmarksConnections{{
    {0, 4},    // (nose, right_eye_inner)
    {4, 5},    // (right_eye_inner, right_eye)
    {5, 6},    // (right_eye, right_eye_outer)
    {6, 8},    // (right_eye_outer, right_ear)
    {0, 1},    // (nose, left_eye_inner)
    {1, 2},    // (left_eye_inner, left_eye)
    {2, 3},    // (left_eye, left_eye_outer)
    {3, 7},    // (left_eye_outer, left_ear)
    {10, 9},   // (mouth_right, mouth_left)
    {12, 11},  // (right_shoulder, left_shoulder)
    {12, 14},  // (right_shoulder, right_elbow)
    {14, 16},  // (right_elbow, right_wrist)
    {16, 18},  // (right_wrist, right_pinky_1)
    {16, 20},  // (right_wrist, right_index_1)
    {16, 22},  // (right_wrist, right_thumb_2)
    {18, 20},  // (right_pinky_1, right_index_1)
    {11, 13},  // (left_shoulder, left_elbow)
    {13, 15},  // (left_elbow, left_wrist)
    {15, 17},  // (left_wrist, left_pinky_1)
    {15, 19},  // (left_wrist, left_index_1)
    {15, 21},  // (left_wrist, left_thumb_2)
    {17, 19},  // (left_pinky_1, left_index_1)
    {12, 24},  // (right_shoulder, right_hip)
    {11, 23},  // (left_shoulder, left_hip)
    {24, 23},  // (right_hip, left_hip)
    {24, 26},  // (right_hip, right_knee)
    {23, 25},  // (left_hip, left_knee)
    {26, 28},  // (right_knee, right_ankle)
    {25, 27},  // (left_knee, left_ankle)
    {28, 30},  // (right_ankle, right_heel)
    {27, 29},  // (left_ankle, left_heel)
    {30, 32},  // (right_heel, right_foot_index)
    {29, 31},  // (left_heel, left_foot_index)
    {28, 32},  // (right_ankle, right_foot_index)
    {27, 31},  // (left_ankle, left_foot_index)
}};

}  // namespace pose_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_LANDMARKS_CONNECTIONS_H_
