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

#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"

#include <optional>
#include <vector>

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

TEST(ConvertFromProto, Succeeds) {
  mediapipe::ClassificationList classification_list_proto;
  mediapipe::Classification& classification_proto =
      *classification_list_proto.add_classification();
  classification_proto.set_index(1);
  classification_proto.set_score(0.5);
  classification_proto.set_label("Left");
  classification_proto.set_display_name("Left_Hand");

  mediapipe::NormalizedLandmarkList normalized_landmark_list_proto;
  mediapipe::NormalizedLandmark& normalized_landmark_proto =
      *normalized_landmark_list_proto.add_landmark();
  normalized_landmark_proto.set_x(0.1);
  normalized_landmark_proto.set_y(0.2);
  normalized_landmark_proto.set_z(0.3);

  mediapipe::LandmarkList landmark_list_proto;
  mediapipe::Landmark& landmark_proto = *landmark_list_proto.add_landmark();
  landmark_proto.set_x(3.1);
  landmark_proto.set_y(5.2);
  landmark_proto.set_z(4.3);

  std::vector<mediapipe::ClassificationList> classification_lists = {
      classification_list_proto};
  std::vector<mediapipe::NormalizedLandmarkList> normalized_landmarks_lists = {
      normalized_landmark_list_proto};
  std::vector<mediapipe::LandmarkList> landmarks_lists = {landmark_list_proto};

  HandLandmarkerResult hand_landmarker_result = ConvertToHandLandmarkerResult(
      classification_lists, normalized_landmarks_lists, landmarks_lists);

  EXPECT_EQ(hand_landmarker_result.handedness.size(), 1);
  EXPECT_EQ(hand_landmarker_result.handedness[0].categories.size(), 1);
  EXPECT_THAT(
      hand_landmarker_result.handedness[0].categories[0],
      testing::FieldsAre(1, testing::FloatEq(0.5), "Left", "Left_Hand"));

  EXPECT_EQ(hand_landmarker_result.hand_landmarks.size(), 1);
  EXPECT_EQ(hand_landmarker_result.hand_landmarks[0].landmarks.size(), 1);
  EXPECT_THAT(hand_landmarker_result.hand_landmarks[0].landmarks[0],
              testing::FieldsAre(testing::FloatEq(0.1), testing::FloatEq(0.2),
                                 testing::FloatEq(0.3), std::nullopt,
                                 std::nullopt, std::nullopt));

  EXPECT_EQ(hand_landmarker_result.hand_world_landmarks.size(), 1);
  EXPECT_EQ(hand_landmarker_result.hand_world_landmarks[0].landmarks.size(), 1);
  EXPECT_THAT(hand_landmarker_result.hand_world_landmarks[0].landmarks[0],
              testing::FieldsAre(testing::FloatEq(3.1), testing::FloatEq(5.2),
                                 testing::FloatEq(4.3), std::nullopt,
                                 std::nullopt, std::nullopt));
}

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
