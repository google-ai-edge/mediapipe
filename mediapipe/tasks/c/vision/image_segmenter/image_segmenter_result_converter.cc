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

#include <algorithm>
#include <cstddef>
#include <vector>

#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter_result.h"

namespace mediapipe::tasks::c::components::containers {

void CppConvertToImageSegmenterResult(
    const mediapipe::tasks::vision::image_segmenter::ImageSegmenterResult& in,
    ImageSegmenterResult* out) {
  // Convert confidence_masks
  if (in.confidence_masks.has_value()) {
    out->confidence_masks_count = in.confidence_masks->size();
    out->confidence_masks = new MpMask[out->confidence_masks_count];
    for (size_t i = 0; i < out->confidence_masks_count; ++i) {
      const auto& image_frame =
          in.confidence_masks.value()[i].GetImageFrameSharedPtr();
      MpMask mp_mask = {
          .type = MpMask::IMAGE_FRAME,
          .image_frame = {.mask_format = MaskFormat::FLOAT,
                          .image_buffer = image_frame->PixelData(),
                          .width = image_frame->Width(),
                          .height = image_frame->Height()}};
      out->confidence_masks[i] = mp_mask;
    }
    out->has_confidence_masks = 1;
  } else {
    out->confidence_masks = nullptr;
    out->confidence_masks_count = 0;
    out->has_confidence_masks = 0;
  }

  // Convert category_mask
  if (in.category_mask.has_value()) {
    const auto& image_frame = in.category_mask.value().GetImageFrameSharedPtr();
    MpMask mp_mask = {.type = MpMask::IMAGE_FRAME,
                      .image_frame = {.mask_format = MaskFormat::UINT8,
                                      .image_buffer = image_frame->PixelData(),
                                      .width = image_frame->Width(),
                                      .height = image_frame->Height()}};
    out->category_mask = mp_mask;
    out->has_category_mask = 1;
  } else {
    out->has_category_mask = 0;
  }

  // Convert quality_scores
  out->quality_scores_count = in.quality_scores.size();
  out->quality_scores = new float[out->quality_scores_count];
  std::copy(in.quality_scores.begin(), in.quality_scores.end(),
            out->quality_scores);
}

void CppCloseImageSegmenterResult(ImageSegmenterResult* result) {
  if (result->has_confidence_masks) {
    delete[] result->confidence_masks;
    result->confidence_masks = nullptr;
    result->confidence_masks_count = 0;
    result->has_confidence_masks = 0;
  }

  if (result->has_category_mask) {
    result->category_mask = {};
    result->has_category_mask = 0;
  }

  delete[] result->quality_scores;
  result->quality_scores = nullptr;
  result->quality_scores_count = 0;
}

}  // namespace mediapipe::tasks::c::components::containers
