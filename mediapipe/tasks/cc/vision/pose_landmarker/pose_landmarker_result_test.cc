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

#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_result.h"

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
namespace pose_landmarker {

TEST(ConvertFromProto, Succeeds) {
  Image segmentation_mask;

  mediapipe::NormalizedLandmarkList normalized_landmark_list_proto;
  mediapipe::NormalizedLandmark& normalized_landmark_proto =
      *normalized_landmark_list_proto.add_landmark();
  normalized_landmark_proto.set_x(0.1);
  normalized_landmark_proto.set_y(0.2);
  normalized_landmark_proto.set_z(0.3);

  mediapipe::LandmarkList world_landmark_list_proto;
  mediapipe::Landmark& landmark_proto =
      *world_landmark_list_proto.add_landmark();
  landmark_proto.set_x(3.1);
  landmark_proto.set_y(5.2);
  landmark_proto.set_z(4.3);

  std::vector<Image> segmentation_masks_lists = {segmentation_mask};

  std::vector<mediapipe::NormalizedLandmarkList> normalized_landmarks_lists = {
      normalized_landmark_list_proto};

  std::vector<mediapipe::LandmarkList> world_landmarks_lists = {
      world_landmark_list_proto};

  PoseLandmarkerResult pose_landmarker_result = ConvertToPoseLandmarkerResult(
      segmentation_masks_lists, normalized_landmarks_lists,
      world_landmarks_lists);

  EXPECT_EQ(pose_landmarker_result.pose_landmarks.size(), 1);
  EXPECT_EQ(pose_landmarker_result.pose_landmarks[0].landmarks.size(), 1);
  EXPECT_THAT(pose_landmarker_result.pose_landmarks[0].landmarks[0],
              testing::FieldsAre(testing::FloatEq(0.1), testing::FloatEq(0.2),
                                 testing::FloatEq(0.3), std::nullopt,
                                 std::nullopt, std::nullopt));

  EXPECT_EQ(pose_landmarker_result.pose_world_landmarks.size(), 1);
  EXPECT_EQ(pose_landmarker_result.pose_world_landmarks[0].landmarks.size(), 1);
  EXPECT_THAT(pose_landmarker_result.pose_world_landmarks[0].landmarks[0],
              testing::FieldsAre(testing::FloatEq(3.1), testing::FloatEq(5.2),
                                 testing::FloatEq(4.3), std::nullopt,
                                 std::nullopt, std::nullopt));
}

}  // namespace pose_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
