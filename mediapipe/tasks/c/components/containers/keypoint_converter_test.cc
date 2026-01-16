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

#include "mediapipe/tasks/c/components/containers/keypoint_converter.h"

#include <string>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/cc/components/containers/keypoint.h"

namespace mediapipe::tasks::c::components::containers {

constexpr float kPrecision = 1e-6;

TEST(KeypointConverterTest, ConvertsKeypointCustomValues) {
  mediapipe::tasks::components::containers::NormalizedKeypoint cpp_keypoint = {
      0.1, 0.2, "foo", 0.5};

  NormalizedKeypoint c_keypoint;
  CppConvertToNormalizedKeypoint(cpp_keypoint, &c_keypoint);
  EXPECT_NEAR(c_keypoint.x, 0.1f, kPrecision);
  EXPECT_NEAR(c_keypoint.y, 0.2f, kPrecision);
  EXPECT_EQ(std::string(c_keypoint.label), "foo");
  EXPECT_NEAR(c_keypoint.score, 0.5f, kPrecision);
  CppCloseNormalizedKeypoint(&c_keypoint);
}

TEST(KeypointConverterTest, FreesMemory) {
  mediapipe::tasks::components::containers::NormalizedKeypoint cpp_keypoint = {
      0.1, 0.2, "foo", 0.5};

  NormalizedKeypoint c_keypoint;
  CppConvertToNormalizedKeypoint(cpp_keypoint, &c_keypoint);
  EXPECT_NE(c_keypoint.label, nullptr);
  CppCloseNormalizedKeypoint(&c_keypoint);
  EXPECT_EQ(c_keypoint.label, nullptr);
}

}  // namespace mediapipe::tasks::c::components::containers
