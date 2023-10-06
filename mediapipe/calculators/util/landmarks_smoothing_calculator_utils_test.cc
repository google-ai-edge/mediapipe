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

#include "mediapipe/calculators/util/landmarks_smoothing_calculator_utils.h"

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace landmarks_smoothing {
namespace {

TEST(LandmarksSmoothingCalculatorUtilsTest, NormalizedLandmarksToLandmarks) {
  NormalizedLandmarkList norm_landmarks;
  NormalizedLandmark* norm_landmark = norm_landmarks.add_landmark();
  norm_landmark->set_x(0.1);
  norm_landmark->set_y(0.2);
  norm_landmark->set_z(0.3);
  norm_landmark->set_visibility(0.4);
  norm_landmark->set_presence(0.5);

  LandmarkList landmarks;
  NormalizedLandmarksToLandmarks(norm_landmarks, /*image_width=*/10,
                                 /*image_height=*/10, landmarks);

  EXPECT_EQ(landmarks.landmark_size(), 1);
  Landmark landmark = landmarks.landmark(0);
  EXPECT_NEAR(landmark.x(), 1.0, 1e-6);
  EXPECT_NEAR(landmark.y(), 2.0, 1e-6);
  EXPECT_NEAR(landmark.z(), 3.0, 1e-6);
  EXPECT_NEAR(landmark.visibility(), 0.4, 1e-6);
  EXPECT_NEAR(landmark.presence(), 0.5, 1e-6);
}

TEST(LandmarksSmoothingCalculatorUtilsTest,
     NormalizedLandmarksToLandmarks_EmptyVisibilityAndPresence) {
  NormalizedLandmarkList norm_landmarks;
  NormalizedLandmark* norm_landmark = norm_landmarks.add_landmark();
  norm_landmark->set_x(0.1);
  norm_landmark->set_y(0.2);
  norm_landmark->set_z(0.3);
  norm_landmark->clear_visibility();
  norm_landmark->clear_presence();

  LandmarkList landmarks;
  NormalizedLandmarksToLandmarks(norm_landmarks, /*image_width=*/10,
                                 /*image_height=*/10, landmarks);

  EXPECT_EQ(landmarks.landmark_size(), 1);
  Landmark landmark = landmarks.landmark(0);
  EXPECT_NEAR(landmark.x(), 1.0, 1e-6);
  EXPECT_NEAR(landmark.y(), 2.0, 1e-6);
  EXPECT_NEAR(landmark.z(), 3.0, 1e-6);
  EXPECT_FALSE(landmark.has_visibility());
  EXPECT_FALSE(landmark.has_presence());
}

TEST(LandmarksSmoothingCalculatorUtilsTest, LandmarksToNormalizedLandmarks) {
  LandmarkList landmarks;
  Landmark* landmark = landmarks.add_landmark();
  landmark->set_x(1.0);
  landmark->set_y(2.0);
  landmark->set_z(3.0);
  landmark->set_visibility(0.4);
  landmark->set_presence(0.5);

  NormalizedLandmarkList norm_landmarks;
  LandmarksToNormalizedLandmarks(landmarks, /*image_width=*/10,
                                 /*image_height=*/10, norm_landmarks);

  EXPECT_EQ(norm_landmarks.landmark_size(), 1);
  NormalizedLandmark norm_landmark = norm_landmarks.landmark(0);
  EXPECT_NEAR(norm_landmark.x(), 0.1, 1e-6);
  EXPECT_NEAR(norm_landmark.y(), 0.2, 1e-6);
  EXPECT_NEAR(norm_landmark.z(), 0.3, 1e-6);
  EXPECT_NEAR(norm_landmark.visibility(), 0.4, 1e-6);
  EXPECT_NEAR(norm_landmark.presence(), 0.5, 1e-6);
}

TEST(LandmarksSmoothingCalculatorUtilsTest,
     LandmarksToNormalizedLandmarks_EmptyVisibilityAndPresence) {
  LandmarkList landmarks;
  Landmark* landmark = landmarks.add_landmark();
  landmark->set_x(1.0);
  landmark->set_y(2.0);
  landmark->set_z(3.0);
  landmark->clear_visibility();
  landmark->clear_presence();

  NormalizedLandmarkList norm_landmarks;
  LandmarksToNormalizedLandmarks(landmarks, /*image_width=*/10,
                                 /*image_height=*/10, norm_landmarks);

  EXPECT_EQ(norm_landmarks.landmark_size(), 1);
  NormalizedLandmark norm_landmark = norm_landmarks.landmark(0);
  EXPECT_NEAR(norm_landmark.x(), 0.1, 1e-6);
  EXPECT_NEAR(norm_landmark.y(), 0.2, 1e-6);
  EXPECT_NEAR(norm_landmark.z(), 0.3, 1e-6);
  EXPECT_FALSE(norm_landmark.has_visibility());
  EXPECT_FALSE(norm_landmark.has_presence());
}

}  // namespace
}  // namespace landmarks_smoothing
}  // namespace mediapipe
