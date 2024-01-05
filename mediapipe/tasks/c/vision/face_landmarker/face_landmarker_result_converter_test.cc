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

#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker_result_converter.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker_result.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"

namespace mediapipe::tasks::c::components::containers {

void InitFaceLandmarkerResult(
    ::mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult*
        cpp_result) {
  // Initialize face_landmarks
  mediapipe::tasks::components::containers::NormalizedLandmark
      cpp_normalized_landmark = {/* x= */ 0.1f, /* y= */ 0.2f, /* z= */ 0.3f};
  mediapipe::tasks::components::containers::NormalizedLandmarks
      cpp_normalized_landmarks;
  cpp_normalized_landmarks.landmarks.push_back(cpp_normalized_landmark);
  cpp_result->face_landmarks.push_back(cpp_normalized_landmarks);

  // Initialize face_blendshapes
  mediapipe::tasks::components::containers::Category cpp_category = {
      /* index= */ 1,
      /* score= */ 0.8f,
      /* category_name= */ "blendshape_label_1",
      /* display_name= */ "blendshape_display_name_1"};
  mediapipe::tasks::components::containers::Classifications
      classifications_for_blendshapes;
  classifications_for_blendshapes.categories.push_back(cpp_category);

  cpp_result->face_blendshapes =
      std::vector<mediapipe::tasks::components::containers::Classifications>{
          classifications_for_blendshapes};
  cpp_result->face_blendshapes->push_back(classifications_for_blendshapes);

  // Initialize facial_transformation_matrixes
  Eigen::MatrixXf cpp_matrix(2, 2);
  cpp_matrix << 1.0f, 2.0f, 3.0f, 4.0f;
  cpp_result->facial_transformation_matrixes = std::vector<Matrix>{cpp_matrix};
}

TEST(FaceLandmarkerResultConverterTest, ConvertsCustomResult) {
  // Initialize a C++ FaceLandmarkerResult
  ::mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult cpp_result;
  InitFaceLandmarkerResult(&cpp_result);

  FaceLandmarkerResult c_result;
  CppConvertToFaceLandmarkerResult(cpp_result, &c_result);

  // Verify conversion of face_landmarks
  EXPECT_EQ(c_result.face_landmarks_count, cpp_result.face_landmarks.size());
  for (uint32_t i = 0; i < c_result.face_landmarks_count; ++i) {
    EXPECT_EQ(c_result.face_landmarks[i].landmarks_count,
              cpp_result.face_landmarks[i].landmarks.size());
    for (uint32_t j = 0; j < c_result.face_landmarks[i].landmarks_count; ++j) {
      const auto& cpp_landmark = cpp_result.face_landmarks[i].landmarks[j];
      EXPECT_FLOAT_EQ(c_result.face_landmarks[i].landmarks[j].x,
                      cpp_landmark.x);
      EXPECT_FLOAT_EQ(c_result.face_landmarks[i].landmarks[j].y,
                      cpp_landmark.y);
      EXPECT_FLOAT_EQ(c_result.face_landmarks[i].landmarks[j].z,
                      cpp_landmark.z);
    }
  }

  // Verify conversion of face_blendshapes
  EXPECT_EQ(c_result.face_blendshapes_count,
            cpp_result.face_blendshapes.value().size());
  for (uint32_t i = 0; i < c_result.face_blendshapes_count; ++i) {
    const auto& cpp_face_blendshapes = cpp_result.face_blendshapes.value();
    EXPECT_EQ(c_result.face_blendshapes[i].categories_count,
              cpp_face_blendshapes[i].categories.size());
    for (uint32_t j = 0; j < c_result.face_blendshapes[i].categories_count;
         ++j) {
      const auto& cpp_category = cpp_face_blendshapes[i].categories[j];
      EXPECT_EQ(c_result.face_blendshapes[i].categories[j].index,
                cpp_category.index);
      EXPECT_FLOAT_EQ(c_result.face_blendshapes[i].categories[j].score,
                      cpp_category.score);
      EXPECT_EQ(
          std::string(c_result.face_blendshapes[i].categories[j].category_name),
          cpp_category.category_name);
    }
  }

  // Verify conversion of facial_transformation_matrixes
  EXPECT_EQ(c_result.facial_transformation_matrixes_count,
            cpp_result.facial_transformation_matrixes.value().size());
  for (uint32_t i = 0; i < c_result.facial_transformation_matrixes_count; ++i) {
    const auto& cpp_facial_transformation_matrixes =
        cpp_result.facial_transformation_matrixes.value();
    // Assuming Matrix struct contains data array and dimensions
    const auto& cpp_matrix = cpp_facial_transformation_matrixes[i];
    EXPECT_EQ(c_result.facial_transformation_matrixes[i].rows,
              cpp_matrix.rows());
    EXPECT_EQ(c_result.facial_transformation_matrixes[i].cols,
              cpp_matrix.cols());
    // Check each element of the matrix
    for (int32_t row = 0; row < cpp_matrix.rows(); ++row) {
      for (int32_t col = 0; col < cpp_matrix.cols(); ++col) {
        size_t index = col * cpp_matrix.rows() + row;  // Column-major index
        EXPECT_FLOAT_EQ(c_result.facial_transformation_matrixes[i].data[index],
                        cpp_matrix(row, col));
      }
    }
  }

  CppCloseFaceLandmarkerResult(&c_result);
}

TEST(FaceLandmarkerResultConverterTest, FreesMemory) {
  ::mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult cpp_result;
  InitFaceLandmarkerResult(&cpp_result);

  FaceLandmarkerResult c_result;
  CppConvertToFaceLandmarkerResult(cpp_result, &c_result);

  EXPECT_NE(c_result.face_blendshapes, nullptr);
  EXPECT_NE(c_result.face_landmarks, nullptr);
  EXPECT_NE(c_result.facial_transformation_matrixes, nullptr);

  CppCloseFaceLandmarkerResult(&c_result);

  EXPECT_EQ(c_result.face_blendshapes, nullptr);
  EXPECT_EQ(c_result.face_landmarks, nullptr);
  EXPECT_EQ(c_result.facial_transformation_matrixes, nullptr);
}

}  // namespace mediapipe::tasks::c::components::containers
