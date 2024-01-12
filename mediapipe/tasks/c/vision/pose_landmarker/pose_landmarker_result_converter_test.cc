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

#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker_result_converter.h"

#include <cstdint>

#include "absl/flags/flag.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe::tasks::c::components::containers {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::DecodeImageFromFile;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kMaskImage[] = "segmentation_input_rotation0.jpg";

void InitPoseLandmarkerResult(
    ::mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerResult*
        cpp_result) {
  // Initialize pose_landmarks
  mediapipe::tasks::components::containers::NormalizedLandmark
      cpp_normalized_landmark = {/* x= */ 0.1f,
                                 /* y= */ 0.2f,
                                 /* z= */ 0.3f};
  mediapipe::tasks::components::containers::NormalizedLandmarks
      cpp_normalized_landmarks;
  cpp_normalized_landmarks.landmarks.push_back(cpp_normalized_landmark);
  cpp_result->pose_landmarks.push_back(cpp_normalized_landmarks);

  // Initialize pose_world_landmarks
  mediapipe::tasks::components::containers::Landmark cpp_landmark = {
      /* x= */ 1.0f,
      /* y= */ 1.1f,
      /* z= */ 1.2f};
  mediapipe::tasks::components::containers::Landmarks cpp_landmarks;
  cpp_landmarks.landmarks.push_back(cpp_landmark);
  cpp_result->pose_world_landmarks.push_back(cpp_landmarks);

  // Initialize segmentation_masks
  MP_ASSERT_OK_AND_ASSIGN(
      Image mask_image,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kMaskImage)));

  // Ensure segmentation_masks is instantiated and add the mask_image to it.
  if (!cpp_result->segmentation_masks) {
    cpp_result->segmentation_masks.emplace();
  }
  cpp_result->segmentation_masks->push_back(mask_image);
}

TEST(PoseLandmarkerResultConverterTest, ConvertsCustomResult) {
  ::mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerResult cpp_result;
  InitPoseLandmarkerResult(&cpp_result);

  PoseLandmarkerResult c_result;
  CppConvertToPoseLandmarkerResult(cpp_result, &c_result);

  // Verify conversion of pose_landmarks
  EXPECT_NE(c_result.pose_landmarks, nullptr);
  EXPECT_EQ(c_result.pose_landmarks_count, cpp_result.pose_landmarks.size());

  for (uint32_t i = 0; i < c_result.pose_landmarks_count; ++i) {
    EXPECT_EQ(c_result.pose_landmarks[i].landmarks_count,
              cpp_result.pose_landmarks[i].landmarks.size());
    for (uint32_t j = 0; j < c_result.pose_landmarks[i].landmarks_count; ++j) {
      const auto& landmark = cpp_result.pose_landmarks[i].landmarks[j];
      EXPECT_FLOAT_EQ(c_result.pose_landmarks[i].landmarks[j].x, landmark.x);
      EXPECT_FLOAT_EQ(c_result.pose_landmarks[i].landmarks[j].y, landmark.y);
      EXPECT_FLOAT_EQ(c_result.pose_landmarks[i].landmarks[j].z, landmark.z);
    }
  }

  // Verify conversion of pose_world_landmarks
  EXPECT_NE(c_result.pose_world_landmarks, nullptr);
  EXPECT_EQ(c_result.pose_world_landmarks_count,
            cpp_result.pose_world_landmarks.size());

  for (uint32_t i = 0; i < c_result.pose_landmarks_count; ++i) {
    EXPECT_EQ(c_result.pose_world_landmarks[i].landmarks_count,
              cpp_result.pose_world_landmarks[i].landmarks.size());
    for (uint32_t j = 0; j < c_result.pose_world_landmarks[i].landmarks_count;
         ++j) {
      const auto& landmark = cpp_result.pose_world_landmarks[i].landmarks[j];
      EXPECT_FLOAT_EQ(c_result.pose_world_landmarks[i].landmarks[j].x,
                      landmark.x);
      EXPECT_FLOAT_EQ(c_result.pose_world_landmarks[i].landmarks[j].y,
                      landmark.y);
      EXPECT_FLOAT_EQ(c_result.pose_world_landmarks[i].landmarks[j].z,
                      landmark.z);
    }
  }

  CppClosePoseLandmarkerResult(&c_result);
}

TEST(PoseLandmarkerResultConverterTest, FreesMemory) {
  ::mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerResult cpp_result;
  InitPoseLandmarkerResult(&cpp_result);

  PoseLandmarkerResult c_result;
  CppConvertToPoseLandmarkerResult(cpp_result, &c_result);

  EXPECT_NE(c_result.pose_landmarks, nullptr);
  EXPECT_NE(c_result.pose_world_landmarks, nullptr);
  EXPECT_NE(c_result.segmentation_masks, nullptr);

  CppClosePoseLandmarkerResult(&c_result);

  EXPECT_EQ(c_result.pose_landmarks, nullptr);
  EXPECT_EQ(c_result.pose_world_landmarks, nullptr);
  EXPECT_EQ(c_result.segmentation_masks, nullptr);
}

}  // namespace mediapipe::tasks::c::components::containers
