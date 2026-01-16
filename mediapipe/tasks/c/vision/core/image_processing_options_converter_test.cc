/* Copyright 2025 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/components/containers/rect.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"

namespace mediapipe::tasks::c::vision::core {

namespace {
using ::mediapipe::tasks::components::containers::RectF;
}  // namespace

TEST(ImageprocessingOptionsConverterTest, ConvertsOptionsWithRegionOfInterest) {
  RectF roi = {0.1f, 0.2f, 0.3f, 0.4f};
  ImageProcessingOptions c_options;
  c_options.rotation_degrees = 180;
  c_options.has_region_of_interest = true;
  c_options.region_of_interest.left = roi.left;
  c_options.region_of_interest.top = roi.top;
  c_options.region_of_interest.right = roi.right;
  c_options.region_of_interest.bottom = roi.bottom;

  mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
  CppConvertToImageProcessingOptions(c_options, &cpp_options);

  EXPECT_EQ(cpp_options.rotation_degrees, 180);
  ASSERT_TRUE(cpp_options.region_of_interest.has_value());
  EXPECT_EQ(cpp_options.region_of_interest->left, roi.left);
  EXPECT_EQ(cpp_options.region_of_interest->top, roi.top);
  EXPECT_EQ(cpp_options.region_of_interest->right, roi.right);
  EXPECT_EQ(cpp_options.region_of_interest->bottom, roi.bottom);
}

TEST(ImageprocessingOptionsConverterTest,
     ConvertsOptionsWithoutRegionOfInterest) {
  ImageProcessingOptions c_options;
  c_options.rotation_degrees = 90;
  c_options.has_region_of_interest = false;

  mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
  CppConvertToImageProcessingOptions(c_options, &cpp_options);

  EXPECT_EQ(cpp_options.rotation_degrees, 90);
  EXPECT_FALSE(cpp_options.region_of_interest.has_value());
}

}  // namespace mediapipe::tasks::c::vision::core
