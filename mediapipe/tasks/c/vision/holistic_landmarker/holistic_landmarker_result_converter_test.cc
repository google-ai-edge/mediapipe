/* Copyright 2026 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker_result_converter.h"

#include <memory>
#include <utility>
#include <vector>

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker_result.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_landmarker_result.h"
#include "testing/base/public/gunit.h"

namespace mediapipe::tasks::c::vision::holistic_landmarker {

using CppCategory = ::mediapipe::tasks::components::containers::Category;
using CppLandmark = ::mediapipe::tasks::components::containers::Landmark;
using CppNormalizedLandmark =
    ::mediapipe::tasks::components::containers::NormalizedLandmark;
using CppHolisticLandmarkerResult =
    ::mediapipe::tasks::vision::holistic_landmarker::HolisticLandmarkerResult;

void CreateHolisticLandmarkerResult(CppHolisticLandmarkerResult* cpp_result) {
  // Initialize landmarks
  CppNormalizedLandmark normalized_landmark = {0.1f, 0.2f, 0.3f};
  CppLandmark landmark = {1.1f, 1.2f, 1.3f};

  cpp_result->face_landmarks.landmarks.push_back(normalized_landmark);
  cpp_result->pose_landmarks.landmarks.push_back(normalized_landmark);
  cpp_result->pose_world_landmarks.landmarks.push_back(landmark);
  cpp_result->left_hand_landmarks.landmarks.push_back(normalized_landmark);
  cpp_result->right_hand_landmarks.landmarks.push_back(normalized_landmark);
  cpp_result->left_hand_world_landmarks.landmarks.push_back(landmark);
  cpp_result->right_hand_world_landmarks.landmarks.push_back(landmark);

  // Initialize face_blendshapes
  CppCategory category = {.index = 0,
                          .score = 0.1f,
                          .category_name = "category_name",
                          .display_name = "display_name"};
  std::vector<CppCategory> categories = {category};
  cpp_result->face_blendshapes = categories;

  // Initialize segmentation_masks
  ImageFrame image_frame(ImageFormat::VEC32F1, 2, 2, 1);
  float* data = reinterpret_cast<float*>(image_frame.MutablePixelData());
  data[0] = 0.1f;
  data[1] = 0.2f;
  data[2] = 0.3f;
  data[3] = 0.4f;
  Image mask_image(std::make_shared<ImageFrame>(std::move(image_frame)));
  cpp_result->pose_segmentation_masks = mask_image;
}

void AssertLandmarksEqual(const Landmarks& c_landmarks,
                          const std::vector<CppLandmark>& cpp_landmarks) {
  EXPECT_EQ(c_landmarks.landmarks_count, cpp_landmarks.size());
  for (int i = 0; i < c_landmarks.landmarks_count; ++i) {
    EXPECT_FLOAT_EQ(c_landmarks.landmarks[i].x, cpp_landmarks[i].x);
    EXPECT_FLOAT_EQ(c_landmarks.landmarks[i].y, cpp_landmarks[i].y);
    EXPECT_FLOAT_EQ(c_landmarks.landmarks[i].z, cpp_landmarks[i].z);
  }
}

void AssertNormalizedLandmarksEqual(
    const NormalizedLandmarks& c_landmarks,
    const std::vector<CppNormalizedLandmark>& cpp_landmarks) {
  EXPECT_EQ(c_landmarks.landmarks_count, cpp_landmarks.size());
  for (int i = 0; i < c_landmarks.landmarks_count; ++i) {
    EXPECT_FLOAT_EQ(c_landmarks.landmarks[i].x, cpp_landmarks[i].x);
    EXPECT_FLOAT_EQ(c_landmarks.landmarks[i].y, cpp_landmarks[i].y);
    EXPECT_FLOAT_EQ(c_landmarks.landmarks[i].z, cpp_landmarks[i].z);
  }
}

TEST(HolisticLandmarkerResultConverterTest, ConvertsCustomResult) {
  CppHolisticLandmarkerResult cpp_result;
  CreateHolisticLandmarkerResult(&cpp_result);

  HolisticLandmarkerResult c_result;
  CppConvertToHolisticLandmarkerResult(cpp_result, &c_result);

  AssertNormalizedLandmarksEqual(c_result.face_landmarks,
                                 cpp_result.face_landmarks.landmarks);
  AssertNormalizedLandmarksEqual(c_result.pose_landmarks,
                                 cpp_result.pose_landmarks.landmarks);
  AssertLandmarksEqual(c_result.pose_world_landmarks,
                       cpp_result.pose_world_landmarks.landmarks);
  AssertNormalizedLandmarksEqual(c_result.left_hand_landmarks,
                                 cpp_result.left_hand_landmarks.landmarks);
  AssertNormalizedLandmarksEqual(c_result.right_hand_landmarks,
                                 cpp_result.right_hand_landmarks.landmarks);
  AssertLandmarksEqual(c_result.left_hand_world_landmarks,
                       cpp_result.left_hand_world_landmarks.landmarks);
  AssertLandmarksEqual(c_result.right_hand_world_landmarks,
                       cpp_result.right_hand_world_landmarks.landmarks);

  EXPECT_EQ(c_result.face_blendshapes.categories_count, 1);
  EXPECT_EQ(c_result.face_blendshapes.categories[0].index, 0);
  EXPECT_FLOAT_EQ(c_result.face_blendshapes.categories[0].score, 0.1f);

  EXPECT_NE(c_result.pose_segmentation_mask, nullptr);
  EXPECT_EQ(c_result.pose_segmentation_mask->image.width(), 2);
  EXPECT_EQ(c_result.pose_segmentation_mask->image.height(), 2);
  const float* data = reinterpret_cast<const float*>(
      c_result.pose_segmentation_mask->image.GetImageFrameSharedPtr()
          ->PixelData());
  EXPECT_FLOAT_EQ(data[0], 0.1f);
  EXPECT_FLOAT_EQ(data[1], 0.2f);
  EXPECT_FLOAT_EQ(data[2], 0.3f);
  EXPECT_FLOAT_EQ(data[3], 0.4f);

  CppCloseHolisticLandmarkerResult(&c_result);
}

TEST(HolisticLandmarkerResultConverterTest, FreesMemory) {
  CppHolisticLandmarkerResult cpp_result;
  CreateHolisticLandmarkerResult(&cpp_result);

  HolisticLandmarkerResult c_result;
  CppConvertToHolisticLandmarkerResult(cpp_result, &c_result);

  EXPECT_NE(c_result.face_landmarks.landmarks, nullptr);
  EXPECT_NE(c_result.pose_landmarks.landmarks, nullptr);
  EXPECT_NE(c_result.pose_world_landmarks.landmarks, nullptr);
  EXPECT_NE(c_result.left_hand_landmarks.landmarks, nullptr);
  EXPECT_NE(c_result.right_hand_landmarks.landmarks, nullptr);
  EXPECT_NE(c_result.left_hand_world_landmarks.landmarks, nullptr);
  EXPECT_NE(c_result.right_hand_world_landmarks.landmarks, nullptr);
  EXPECT_NE(c_result.face_blendshapes.categories, nullptr);
  EXPECT_NE(c_result.pose_segmentation_mask, nullptr);

  CppCloseHolisticLandmarkerResult(&c_result);

  EXPECT_EQ(c_result.face_landmarks.landmarks, nullptr);
  EXPECT_EQ(c_result.pose_landmarks.landmarks, nullptr);
  EXPECT_EQ(c_result.pose_world_landmarks.landmarks, nullptr);
  EXPECT_EQ(c_result.left_hand_landmarks.landmarks, nullptr);
  EXPECT_EQ(c_result.right_hand_landmarks.landmarks, nullptr);
  EXPECT_EQ(c_result.left_hand_world_landmarks.landmarks, nullptr);
  EXPECT_EQ(c_result.right_hand_world_landmarks.landmarks, nullptr);
  EXPECT_EQ(c_result.face_blendshapes.categories, nullptr);
  EXPECT_EQ(c_result.pose_segmentation_mask, nullptr);
}

}  // namespace mediapipe::tasks::c::vision::holistic_landmarker
