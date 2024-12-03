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

#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result_converter.h"

#include <cstdint>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"

namespace mediapipe::tasks::c::components::containers {

void InitHandLandmarkerResult(
    ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult*
        cpp_result) {
  // Initialize handedness
  mediapipe::tasks::components::containers::Category cpp_category = {
      /* index= */ 1,
      /* score= */ 0.8f,
      /* category_name= */ "handeness_label_1",
      /* display_name= */ "handeness_display_name_1"};
  mediapipe::tasks::components::containers::Classifications
      classifications_for_handedness;
  classifications_for_handedness.categories.push_back(cpp_category);
  cpp_result->handedness.push_back(classifications_for_handedness);

  // Initialize hand_landmarks
  mediapipe::tasks::components::containers::NormalizedLandmark
      cpp_normalized_landmark = {/* x= */ 0.1f,
                                 /* y= */ 0.2f,
                                 /* z= */ 0.3f};
  mediapipe::tasks::components::containers::NormalizedLandmarks
      cpp_normalized_landmarks;
  cpp_normalized_landmarks.landmarks.push_back(cpp_normalized_landmark);
  cpp_result->hand_landmarks.push_back(cpp_normalized_landmarks);

  // Initialize hand_world_landmarks
  mediapipe::tasks::components::containers::Landmark cpp_landmark = {
      /* x= */ 1.0f,
      /* y= */ 1.1f,
      /* z= */ 1.2f};
  mediapipe::tasks::components::containers::Landmarks cpp_landmarks;
  cpp_landmarks.landmarks.push_back(cpp_landmark);
  cpp_result->hand_world_landmarks.push_back(cpp_landmarks);
}

TEST(HandLandmarkerResultConverterTest, ConvertsCustomResult) {
  ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult cpp_result;
  InitHandLandmarkerResult(&cpp_result);

  HandLandmarkerResult c_result;
  CppConvertToHandLandmarkerResult(cpp_result, &c_result);

  // Verify conversion of hand_landmarks
  EXPECT_NE(c_result.hand_landmarks, nullptr);
  EXPECT_EQ(c_result.hand_landmarks_count, cpp_result.hand_landmarks.size());

  for (uint32_t i = 0; i < c_result.hand_landmarks_count; ++i) {
    EXPECT_EQ(c_result.hand_landmarks[i].landmarks_count,
              cpp_result.hand_landmarks[i].landmarks.size());
    for (uint32_t j = 0; j < c_result.hand_landmarks[i].landmarks_count; ++j) {
      const auto& landmark = cpp_result.hand_landmarks[i].landmarks[j];
      EXPECT_FLOAT_EQ(c_result.hand_landmarks[i].landmarks[j].x, landmark.x);
      EXPECT_FLOAT_EQ(c_result.hand_landmarks[i].landmarks[j].y, landmark.y);
      EXPECT_FLOAT_EQ(c_result.hand_landmarks[i].landmarks[j].z, landmark.z);
    }
  }

  // Verify conversion of hand_world_landmarks
  EXPECT_NE(c_result.hand_world_landmarks, nullptr);
  EXPECT_EQ(c_result.hand_world_landmarks_count,
            cpp_result.hand_world_landmarks.size());

  for (uint32_t i = 0; i < c_result.hand_landmarks_count; ++i) {
    EXPECT_EQ(c_result.hand_world_landmarks[i].landmarks_count,
              cpp_result.hand_world_landmarks[i].landmarks.size());
    for (uint32_t j = 0; j < c_result.hand_world_landmarks[i].landmarks_count;
         ++j) {
      const auto& landmark = cpp_result.hand_world_landmarks[i].landmarks[j];
      EXPECT_FLOAT_EQ(c_result.hand_world_landmarks[i].landmarks[j].x,
                      landmark.x);
      EXPECT_FLOAT_EQ(c_result.hand_world_landmarks[i].landmarks[j].y,
                      landmark.y);
      EXPECT_FLOAT_EQ(c_result.hand_world_landmarks[i].landmarks[j].z,
                      landmark.z);
    }
  }

  CppCloseHandLandmarkerResult(&c_result);
}

TEST(HandLandmarkerResultConverterTest, FreesMemory) {
  ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult cpp_result;
  InitHandLandmarkerResult(&cpp_result);

  HandLandmarkerResult c_result;
  CppConvertToHandLandmarkerResult(cpp_result, &c_result);

  EXPECT_NE(c_result.handedness, nullptr);
  EXPECT_NE(c_result.hand_landmarks, nullptr);
  EXPECT_NE(c_result.hand_world_landmarks, nullptr);

  CppCloseHandLandmarkerResult(&c_result);

  EXPECT_EQ(c_result.handedness, nullptr);
  EXPECT_EQ(c_result.hand_landmarks, nullptr);
  EXPECT_EQ(c_result.hand_world_landmarks, nullptr);
}

}  // namespace mediapipe::tasks::c::components::containers
