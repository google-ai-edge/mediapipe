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

#include "mediapipe/tasks/c/test/test_utils.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/vision/core/common.h"

namespace mediapipe::tasks::c::test {

MpMask CreateCategoryMaskFromImage(absl::StatusOr<Image>& image) {
  const auto& image_frame = image->GetImageFrameSharedPtr();

  const int pixel_data_size = image_frame->PixelDataSizeStoredContiguously();
  auto* pixel_data = new uint8_t[pixel_data_size];
  image_frame->CopyToBuffer(pixel_data, pixel_data_size);

  MpMask mask = {.type = MpMask::IMAGE_FRAME,
                 .image_frame = {.mask_format = MaskFormat::UINT8,
                                 .image_buffer = pixel_data,
                                 .width = image_frame->Width(),
                                 .height = image_frame->Height()}};

  return mask;
}

float SimilarToUint8Mask(const MpMask* actual_mask, const MpMask* expected_mask,
                         int magnification_factor) {
  // Validate that both images are of the same size and type
  if (actual_mask->image_frame.width != expected_mask->image_frame.width ||
      actual_mask->image_frame.height != expected_mask->image_frame.height ||
      actual_mask->image_frame.mask_format != MaskFormat::UINT8 ||
      expected_mask->image_frame.mask_format != MaskFormat::UINT8) {
    return 0;  // Not similar
  }

  int consistent_pixels = 0;
  int total_pixels =
      actual_mask->image_frame.width * actual_mask->image_frame.height;

  const uint8_t* buffer_actual = actual_mask->image_frame.image_buffer;
  const uint8_t* buffer_expected = expected_mask->image_frame.image_buffer;

  for (int i = 0; i < total_pixels; ++i) {
    // Apply magnification factor and compare
    if (buffer_actual[i] * magnification_factor == buffer_expected[i]) {
      ++consistent_pixels;
    }
  }

  float similarity = (float)consistent_pixels / total_pixels;
  return similarity;
}

}  // namespace mediapipe::tasks::c::test
