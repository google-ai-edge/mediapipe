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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKS_CONNECTIONS_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKS_CONNECTIONS_H_

#include <array>

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

static constexpr std::array<std::array<int, 2>, 6> kHandPalmConnections{
    {{0, 1}, {0, 5}, {9, 13}, {13, 17}, {5, 9}, {0, 17}}};

static constexpr std::array<std::array<int, 2>, 3> kHandThumbConnections{
    {{1, 2}, {2, 3}, {3, 4}}};

static constexpr std::array<std::array<int, 2>, 3> kHandIndexFingerConnections{
    {{5, 6}, {6, 7}, {7, 8}}};

static constexpr std::array<std::array<int, 2>, 3> kHandMiddleFingerConnections{
    {{9, 10}, {10, 11}, {11, 12}}};

static constexpr std::array<std::array<int, 2>, 3> kHandRingFingerConnections{
    {{13, 14}, {14, 15}, {15, 16}}};

static constexpr std::array<std::array<int, 2>, 3> kHandPinkyFingerConnections{
    {{17, 18}, {18, 19}, {19, 20}}};

static constexpr std::array<std::array<int, 2>, 21> kHandConnections{
    {{0, 1},   {0, 5},   {9, 13},  {13, 17}, {5, 9},   {0, 17},  {1, 2},
     {2, 3},   {3, 4},   {5, 6},   {6, 7},   {7, 8},   {9, 10},  {10, 11},
     {11, 12}, {13, 14}, {14, 15}, {15, 16}, {17, 18}, {18, 19}, {19, 20}}};

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKS_CONNECTIONS_H_
