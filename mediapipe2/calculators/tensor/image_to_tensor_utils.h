// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_IMAGE_TO_TENSOR_UTILS_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_IMAGE_TO_TENSOR_UTILS_H_

#include <array>

#include "absl/types/optional.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {

struct RotatedRect {
  float center_x;
  float center_y;
  float width;
  float height;
  float rotation;
};

// Generates a new ROI or converts it from normalized rect.
RotatedRect GetRoi(int input_width, int input_height,
                   absl::optional<mediapipe::NormalizedRect> norm_rect);

// Pads ROI, so extraction happens correctly if aspect ratio is to be kept.
// Returns letterbox padding applied.
absl::StatusOr<std::array<float, 4>> PadRoi(int input_tensor_width,
                                            int input_tensor_height,
                                            bool keep_aspect_ratio,
                                            RotatedRect* roi);

// Represents a transformation of value which involves scaling and offsetting.
// To apply transformation:
// ValueTransformation transform = ...
// float transformed_value = transform.scale * value + transfrom.offset;
struct ValueTransformation {
  float scale;
  float offset;
};

// Returns value transformation to apply to a value in order to convert it from
// [from_range_min, from_range_max] into [to_range_min, to_range_max] range.
// from_range_min must be less than from_range_max
// to_range_min must be less than to_range_max
absl::StatusOr<ValueTransformation> GetValueRangeTransformation(
    float from_range_min, float from_range_max, float to_range_min,
    float to_range_max);

// Populates 4x4 "matrix" with row major order transformation matrix which
// maps (x, y) in range [0, 1] (describing points of @sub_rect)
// to (x', y') in range [0, 1]*** (describing points of a rect:
// [0, @rect_width] x [0, @rect_height] = RECT).
//
// *** (x', y') will go out of the range for points from @sub_rect
//     which are not contained by RECT and it's expected behavior
//
// @sub_rect - rotated sub rect in absolute coordinates
// @rect_width - rect width
// @rect_height - rect height
// @flip_horizontaly - we need to flip the output buffer.
// @matrix - 4x4 matrix (array of 16 elements) to populate
void GetRotatedSubRectToRectTransformMatrix(const RotatedRect& sub_rect,
                                            int rect_width, int rect_height,
                                            bool flip_horizontaly,
                                            std::array<float, 16>* matrix);

// Returns the transpose of the matrix found with
// "GetRotatedSubRectToRectTransformMatrix".  That is to say, this populates a
// 4x4 "matrix" with col major order transformation matrix which maps (x, y) in
// range [0, 1] (describing points of @sub_rect) to (x', y') in range [0, 1]***
// (describing points of a rect: [0, @rect_width] x [0, @rect_height] = RECT).
//
// *** (x', y') will go out of the range for points from @sub_rect
//     which are not contained by RECT and it's expected behavior
//
// @sub_rect - rotated sub rect in absolute coordinates
// @rect_width - rect width
// @rect_height - rect height
// @flip_horizontaly - we need to flip the output buffer.
// @matrix - 4x4 matrix (array of 16 elements) to populate
void GetTransposedRotatedSubRectToRectTransformMatrix(
    const RotatedRect& sub_rect, int rect_width, int rect_height,
    bool flip_horizontaly, std::array<float, 16>* matrix);

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_IMAGE_TO_TENSOR_UTILS_H_
