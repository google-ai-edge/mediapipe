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

#include "mediapipe/tasks/cc/vision/hand_gesture_recognizer/handedness_util.h"

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace {

TEST(GetLeftHandScore, SingleLeftHandClassification) {
  ClassificationList classifications;
  auto& c = *classifications.add_classification();
  c.set_label("Left");
  c.set_score(0.6f);

  MP_ASSERT_OK_AND_ASSIGN(float score, GetLeftHandScore(classifications));
  EXPECT_FLOAT_EQ(score, 0.6f);
}

TEST(GetLeftHandScore, SingleRightHandClassification) {
  ClassificationList classifications;
  auto& c = *classifications.add_classification();
  c.set_label("Right");
  c.set_score(0.9f);

  MP_ASSERT_OK_AND_ASSIGN(float score, GetLeftHandScore(classifications));
  EXPECT_FLOAT_EQ(score, 0.1f);
}

TEST(GetLeftHandScore, LeftAndRightHandClassification) {
  ClassificationList classifications;
  auto& right = *classifications.add_classification();
  right.set_label("Right");
  right.set_score(0.9f);
  auto& left = *classifications.add_classification();
  left.set_label("Left");
  left.set_score(0.1f);

  MP_ASSERT_OK_AND_ASSIGN(float score, GetLeftHandScore(classifications));
  EXPECT_FLOAT_EQ(score, 0.1f);
}

TEST(GetLeftHandScore, LeftAndRightLowerCaseHandClassification) {
  ClassificationList classifications;
  auto& right = *classifications.add_classification();
  right.set_label("right");
  right.set_score(0.9f);
  auto& left = *classifications.add_classification();
  left.set_label("left");
  left.set_score(0.1f);

  MP_ASSERT_OK_AND_ASSIGN(float score, GetLeftHandScore(classifications));
  EXPECT_FLOAT_EQ(score, 0.1f);
}

}  // namespace
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
