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

#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"

#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

testing::Matcher<RotatedRect> EqRotatedRect(float width, float height,
                                            float center_x, float center_y,
                                            float rotation) {
  return testing::AllOf(
      testing::Field(&RotatedRect::width, testing::FloatEq(width)),
      testing::Field(&RotatedRect::height, testing::FloatEq(height)),
      testing::Field(&RotatedRect::center_x, testing::FloatEq(center_x)),
      testing::Field(&RotatedRect::center_y, testing::FloatEq(center_y)),
      testing::Field(&RotatedRect::rotation, testing::FloatEq(rotation)));
}

TEST(GetRoi, NoNormRect) {
  EXPECT_THAT(GetRoi(4, 4, {}), EqRotatedRect(4, 4, 2, 2, 0));
  EXPECT_THAT(GetRoi(25, 15, {}), EqRotatedRect(25, 15, 12.5f, 7.5f, 0));
}

TEST(GetRoi, WholeImageNormRect) {
  mediapipe::NormalizedRect norm_rect;
  norm_rect.set_width(1.0f);
  norm_rect.set_height(1.0f);
  norm_rect.set_x_center(0.5f);
  norm_rect.set_y_center(0.5f);
  norm_rect.set_rotation(0.0f);
  EXPECT_THAT(GetRoi(4, 4, norm_rect), EqRotatedRect(4, 4, 2, 2, 0));
  EXPECT_THAT(GetRoi(25, 15, norm_rect), EqRotatedRect(25, 15, 12.5f, 7.5f, 0));
}

TEST(GetRoi, ExpandedNormRect) {
  mediapipe::NormalizedRect norm_rect;
  norm_rect.set_width(4.0f);
  norm_rect.set_height(2.0f);
  norm_rect.set_x_center(0.5f);
  norm_rect.set_y_center(1.0f);
  norm_rect.set_rotation(3.0f);
  EXPECT_THAT(GetRoi(4, 4, norm_rect), EqRotatedRect(16, 8, 2, 4, 3));
  EXPECT_THAT(GetRoi(25, 15, norm_rect), EqRotatedRect(100, 30, 12.5f, 15, 3));
}

TEST(PadRoi, NoPadding) {
  RotatedRect roi{.center_x = 20,
                  .center_y = 10,
                  .width = 100,
                  .height = 200,
                  .rotation = 5};
  auto status_or_value = PadRoi(10, 10, /*keep_aspect_ratio=*/false, &roi);
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(),
              ElementsAreArray({0.0f, 0.0f, 0.0f, 0.0f}));
  EXPECT_THAT(roi, EqRotatedRect(100, 200, 20, 10, 5));
}

TEST(PadRoi, HorizontalPadding) {
  RotatedRect roi{.center_x = 20,
                  .center_y = 10,
                  .width = 100,
                  .height = 200,
                  .rotation = 5};
  auto status_or_value = PadRoi(10, 10, /*keep_aspect_ratio=*/true, &roi);
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(),
              ElementsAreArray({0.25f, 0.0f, 0.25f, 0.0f}));
  EXPECT_THAT(roi, EqRotatedRect(200, 200, 20, 10, 5));
}

TEST(PadRoi, VerticalPadding) {
  RotatedRect roi{
      .center_x = 1, .center_y = 2, .width = 21, .height = 19, .rotation = 3};
  const float expected_horizontal_padding = (21 - 19) / 2.0f / 21;
  auto status_or_value = PadRoi(10, 10, /*keep_aspect_ratio=*/true, &roi);
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(
      status_or_value.value(),
      ElementsAre(testing::FloatEq(0.0f),
                  testing::FloatNear(expected_horizontal_padding, 1e-6),
                  testing::FloatEq(0.0f),
                  testing::FloatNear(expected_horizontal_padding, 1e-6)));
  EXPECT_THAT(roi, EqRotatedRect(21, 21, 1, 2, 3));
}

testing::Matcher<ValueTransformation> EqValueTransformation(float scale,
                                                            float offset) {
  return ::testing::AllOf(
      testing::Field(&ValueTransformation::scale, testing::FloatEq(scale)),
      testing::Field(&ValueTransformation::offset, testing::FloatEq(offset)));
}

TEST(GetValueRangeTransformation, PixelToFloatZeroCenter) {
  auto status_or_value = GetValueRangeTransformation(
      /*from_range_min=*/0.0f, /*from_range_max=*/255.0f,
      /*to_range_min=*/-1.0f, /*to_range_max=*/1.0f);
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(),
              EqValueTransformation(/*scale=*/2 / 255.0f,
                                    /*offset=*/-1.0f));
}

TEST(GetValueRangeTransformation, PixelToFloat) {
  auto status_or_value = GetValueRangeTransformation(
      /*from_range_min=*/0.0f, /*from_range_max=*/255.0f,
      /*to_range_min=*/0.0f, /*to_range_max=*/1.0f);
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(),
              EqValueTransformation(/*scale=*/1 / 255.0f,
                                    /*offset=*/0.0f));
}

TEST(GetValueRangeTransformation, FloatToFloatNoOp) {
  auto status_or_value = GetValueRangeTransformation(
      /*from_range_min=*/0.0f, /*from_range_max=*/1.0f,
      /*to_range_min=*/0.0f, /*to_range_max=*/1.0f);
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(),
              EqValueTransformation(/*scale=*/1.0f, /*offset=*/0.0f));
}

TEST(GetValueRangeTransformation, PixelToPixelNoOp) {
  auto status_or_value = GetValueRangeTransformation(
      /*from_range_min=*/0.0f, /*from_range_max=*/255.0f,
      /*to_range_min=*/0.0f, /*to_range_max=*/255.0f);
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(),
              EqValueTransformation(/*scale=*/1.0f, /*offset=*/0.0f));
}

TEST(GetValueRangeTransformation, FloatToPixel) {
  auto status_or_value = GetValueRangeTransformation(
      /*from_range_min=*/0.0f, /*from_range_max=*/1.0f,
      /*to_range_min=*/0.0f, /*to_range_max=*/255.0f);
  MP_ASSERT_OK(status_or_value);
  EXPECT_THAT(status_or_value.value(),
              EqValueTransformation(/*scale=*/255.0f, /*offset=*/0.0f));
}

}  // namespace
}  // namespace mediapipe
