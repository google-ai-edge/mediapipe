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

#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"

#include <optional>
#include <vector>

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_landmarker {

class FaceLandmarkerResultTest : public ::testing::Test {};

TEST(FaceLandmarkerResultTest, Succeeds) {
  mediapipe::NormalizedLandmarkList face_landmarks_proto;
  mediapipe::NormalizedLandmark& normalized_landmark_proto =
      *face_landmarks_proto.add_landmark();
  normalized_landmark_proto.set_x(0.1);
  normalized_landmark_proto.set_y(0.2);
  normalized_landmark_proto.set_z(0.3);

  mediapipe::ClassificationList face_blendshapes_proto;
  mediapipe::Classification& classification_proto =
      *face_blendshapes_proto.add_classification();
  classification_proto.set_index(1);
  classification_proto.set_score(0.9075446);
  classification_proto.set_label("browDownLeft");

  mediapipe::MatrixData matrix_proto;
  matrix_proto.set_rows(3);
  matrix_proto.set_cols(3);
  for (int i = 0; i < matrix_proto.rows() * matrix_proto.cols(); i++) {
    matrix_proto.add_packed_data(i);
  }

  FaceLandmarkerResult face_landmarker_result = ConvertToFaceLandmarkerResult(
      {{face_landmarks_proto}}, {{face_blendshapes_proto}}, {{matrix_proto}});

  EXPECT_EQ(face_landmarker_result.face_landmarks.size(), 1);
  EXPECT_EQ(face_landmarker_result.face_landmarks[0].landmarks.size(), 1);
  EXPECT_THAT(face_landmarker_result.face_landmarks[0].landmarks[0],
              testing::FieldsAre(testing::FloatEq(0.1), testing::FloatEq(0.2),
                                 testing::FloatEq(0.3), std::nullopt,
                                 std::nullopt, std::nullopt));

  ASSERT_TRUE(face_landmarker_result.face_blendshapes.has_value());
  EXPECT_EQ(face_landmarker_result.face_blendshapes->size(), 1);
  EXPECT_EQ(face_landmarker_result.face_blendshapes->at(0).categories.size(),
            1);
  EXPECT_THAT(face_landmarker_result.face_blendshapes->at(0).categories[0],
              testing::FieldsAre(1, testing::FloatEq(0.9075446), "browDownLeft",
                                 std::nullopt));

  Matrix expected_matrix{{0, 3, 6}, {1, 4, 7}, {2, 5, 8}};
  ASSERT_TRUE(
      face_landmarker_result.facial_transformation_matrixes.has_value());
  EXPECT_EQ(face_landmarker_result.facial_transformation_matrixes->size(), 1);
  EXPECT_EQ(face_landmarker_result.facial_transformation_matrixes->at(0),
            expected_matrix);
}

}  // namespace face_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
