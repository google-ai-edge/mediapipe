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

#include "absl/log/absl_log.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"

namespace mediapipe::tasks::c::test {

MpImagePtr CreateCategoryMaskFromImage(const Image& image) {
  const auto& image_frame = image.GetImageFrameSharedPtr();

  const int pixel_data_size = image_frame->PixelDataSizeStoredContiguously();
  const uint8_t* pixel_data = image_frame->PixelData();

  MpImagePtr mp_image;
  char* error_msg = nullptr;
  MpStatus status = MpImageCreateFromUint8Data(
      MpImageFormat::kMpImageFormatGray8, image_frame->Width(),
      image_frame->Height(), pixel_data, pixel_data_size, &mp_image,
      &error_msg);
  if (status != kMpOk) {
    ABSL_LOG(ERROR) << "Failed to create MP Image: " << error_msg;
    return nullptr;
  }
  return mp_image;
}

float SimilarToUint8Mask(MpImageInternal* actual_mask,
                         MpImageInternal* expected_mask,
                         int magnification_factor) {
  // Validate that both images are of the same size and type
  if (MpImageGetWidth(actual_mask) != MpImageGetWidth(expected_mask) ||
      MpImageGetHeight(actual_mask) != MpImageGetHeight(expected_mask) ||
      MpImageGetFormat(actual_mask) != MpImageFormat::kMpImageFormatGray8 ||
      MpImageGetFormat(expected_mask) != MpImageFormat::kMpImageFormatGray8) {
    return 0;  // Not similar
  }

  int consistent_pixels = 0;
  int total_pixels =
      MpImageGetWidth(actual_mask) * MpImageGetHeight(actual_mask);

  const uint8_t* buffer_actual;
  MpImageDataUint8(actual_mask, &buffer_actual, /*error_msg=*/nullptr);
  const uint8_t* buffer_expected;
  MpImageDataUint8(expected_mask, &buffer_expected, /*error_msg=*/nullptr);

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
