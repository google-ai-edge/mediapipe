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

#include "mediapipe/tasks/c/components/containers/landmark_converter.h"

#include <cstdlib>
#include <vector>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"

namespace mediapipe::tasks::c::components::containers {

TEST(LandmarkConverterTest, ConvertsCustomLandmark) {
  mediapipe::tasks::components::containers::Landmark cpp_landmark = {0.1f, 0.2f,
                                                                     0.3f};

  ::Landmark c_landmark;
  CppConvertToLandmark(cpp_landmark, &c_landmark);
  EXPECT_FLOAT_EQ(c_landmark.x, cpp_landmark.x);
  EXPECT_FLOAT_EQ(c_landmark.y, cpp_landmark.y);
  EXPECT_FLOAT_EQ(c_landmark.z, cpp_landmark.z);
  CppCloseLandmark(&c_landmark);
}

TEST(LandmarksConverterTest, ConvertsCustomLandmarks) {
  std::vector<mediapipe::tasks::components::containers::Landmark>
      cpp_landmarks = {
          {0.1f, 0.2f, 0.3f},  // First Landmark
          {0.4f, 0.5f, 0.6f}   // Second Landmark
      };

  ::Landmarks c_landmarks;
  CppConvertToLandmarks(cpp_landmarks, &c_landmarks);

  EXPECT_EQ(c_landmarks.landmarks_count, cpp_landmarks.size());
  for (size_t i = 0; i < c_landmarks.landmarks_count; ++i) {
    EXPECT_FLOAT_EQ(c_landmarks.landmarks[i].x, cpp_landmarks[i].x);
    EXPECT_FLOAT_EQ(c_landmarks.landmarks[i].y, cpp_landmarks[i].y);
    EXPECT_FLOAT_EQ(c_landmarks.landmarks[i].z, cpp_landmarks[i].z);
  }

  CppCloseLandmarks(&c_landmarks);
}

TEST(NormalizedLandmarkConverterTest, ConvertsCustomNormalizedLandmark) {
  mediapipe::tasks::components::containers::NormalizedLandmark
      cpp_normalized_landmark = {0.7f, 0.8f, 0.9f};

  ::NormalizedLandmark c_normalized_landmark;
  CppConvertToNormalizedLandmark(cpp_normalized_landmark,
                                 &c_normalized_landmark);

  EXPECT_FLOAT_EQ(c_normalized_landmark.x, cpp_normalized_landmark.x);
  EXPECT_FLOAT_EQ(c_normalized_landmark.y, cpp_normalized_landmark.y);
  EXPECT_FLOAT_EQ(c_normalized_landmark.z, cpp_normalized_landmark.z);

  CppCloseNormalizedLandmark(&c_normalized_landmark);
}

TEST(NormalizedLandmarksConverterTest, ConvertsCustomNormalizedLandmarks) {
  std::vector<mediapipe::tasks::components::containers::NormalizedLandmark>
      cpp_normalized_landmarks = {
          {0.1f, 0.2f, 0.3f},  // First NormalizedLandmark
          {0.4f, 0.5f, 0.6f}   // Second NormalizedLandmark
      };

  ::NormalizedLandmarks c_normalized_landmarks;
  CppConvertToNormalizedLandmarks(cpp_normalized_landmarks,
                                  &c_normalized_landmarks);

  EXPECT_EQ(c_normalized_landmarks.landmarks_count,
            cpp_normalized_landmarks.size());
  for (size_t i = 0; i < c_normalized_landmarks.landmarks_count; ++i) {
    EXPECT_FLOAT_EQ(c_normalized_landmarks.landmarks[i].x,
                    cpp_normalized_landmarks[i].x);
    EXPECT_FLOAT_EQ(c_normalized_landmarks.landmarks[i].y,
                    cpp_normalized_landmarks[i].y);
    EXPECT_FLOAT_EQ(c_normalized_landmarks.landmarks[i].z,
                    cpp_normalized_landmarks[i].z);
  }

  CppCloseNormalizedLandmarks(&c_normalized_landmarks);
}

TEST(LandmarkConverterTest, FreesMemory) {
  mediapipe::tasks::components::containers::Landmark cpp_landmark = {
      0.1f, 0.2f, 0.3f, 0.0f, 0.0f, "foo"};

  ::Landmark c_landmark;
  CppConvertToLandmark(cpp_landmark, &c_landmark);
  EXPECT_NE(c_landmark.name, nullptr);

  CppCloseLandmark(&c_landmark);
  EXPECT_EQ(c_landmark.name, nullptr);
}

TEST(NormalizedLandmarkConverterTest, FreesMemory) {
  mediapipe::tasks::components::containers::NormalizedLandmark cpp_landmark = {
      0.1f, 0.2f, 0.3f, 0.0f, 0.0f, "foo"};

  ::NormalizedLandmark c_landmark;
  CppConvertToNormalizedLandmark(cpp_landmark, &c_landmark);
  EXPECT_NE(c_landmark.name, nullptr);

  CppCloseNormalizedLandmark(&c_landmark);
  EXPECT_EQ(c_landmark.name, nullptr);
}

TEST(LandmarksConverterTest, FreesMemory) {
  std::vector<mediapipe::tasks::components::containers::Landmark>
      cpp_landmarks = {{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}};

  ::Landmarks c_landmarks;
  CppConvertToLandmarks(cpp_landmarks, &c_landmarks);
  EXPECT_NE(c_landmarks.landmarks, nullptr);

  CppCloseLandmarks(&c_landmarks);
  EXPECT_EQ(c_landmarks.landmarks, nullptr);
}

TEST(NormalizedLandmarksConverterTest, FreesMemory) {
  std::vector<mediapipe::tasks::components::containers::NormalizedLandmark>
      cpp_normalized_landmarks = {{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}};

  ::NormalizedLandmarks c_normalized_landmarks;
  CppConvertToNormalizedLandmarks(cpp_normalized_landmarks,
                                  &c_normalized_landmarks);
  EXPECT_NE(c_normalized_landmarks.landmarks, nullptr);

  CppCloseNormalizedLandmarks(&c_normalized_landmarks);
  EXPECT_EQ(c_normalized_landmarks.landmarks, nullptr);
}

}  // namespace mediapipe::tasks::c::components::containers
