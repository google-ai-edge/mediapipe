/* Copyright 2024 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result_converter.h"

#include "absl/flags/flag.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe::tasks::c::components::containers {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::vision::DecodeImageFromFile;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/vision/";
constexpr char kMaskImage[] = "segmentation_input_rotation0.jpg";

TEST(ImageSegmenterResultConverterTest, ConvertsCategoryMaskAndFreesMemory) {
  mediapipe::tasks::vision::image_segmenter::ImageSegmenterResult cpp_result;

  // Create a mock Image for category_mask
  MP_ASSERT_OK_AND_ASSIGN(
      Image expected_mask,
      DecodeImageFromFile(JoinPath("./", kTestDataDirectory, kMaskImage)));

  // Set the category_mask in cpp_result
  cpp_result.category_mask = expected_mask;

  // Set some sample quality scores
  cpp_result.quality_scores = {0.9f, 0.8f, 0.95f};  // Example quality scores

  // Convert the C++ result to C struct
  ImageSegmenterResult c_result;
  CppConvertToImageSegmenterResult(cpp_result, &c_result);

  // Verify the conversion
  EXPECT_TRUE(c_result.has_category_mask);
  EXPECT_EQ(c_result.category_mask.type, MpMask::IMAGE_FRAME);
  EXPECT_EQ(c_result.category_mask.image_frame.mask_format, MaskFormat::UINT8);
  EXPECT_EQ(c_result.category_mask.image_frame.width, expected_mask.width());
  EXPECT_EQ(c_result.category_mask.image_frame.height, expected_mask.height());

  CppCloseImageSegmenterResult(&c_result);
  EXPECT_EQ(c_result.confidence_masks, nullptr);
  EXPECT_FALSE(c_result.has_category_mask);
  EXPECT_EQ(c_result.quality_scores, nullptr);
}

}  // namespace mediapipe::tasks::c::components::containers
