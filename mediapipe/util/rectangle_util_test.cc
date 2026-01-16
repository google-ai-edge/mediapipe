// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/util/rectangle_util.h"

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::mediapipe::NormalizedRect;
using ::testing::FloatNear;

class RectangleUtilTest : public testing::Test {
 protected:
  RectangleUtilTest() {
    //  0.4                                         ================
    //                                              |    |    |    |
    //  0.3 =====================                   |   NR2   |    |
    //      |    |    |   NR1   |                   |    |    NR4  |
    //  0.2 |   NR0   |    ===========              ================
    //      |    |    |    |    |    |
    //  0.1 =====|===============    |
    //           |    NR3  |    |    |
    //  0.0      ================    |
    //                     |   NR5   |
    // -0.1                ===========
    //     0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0  1.1  1.2

    // NormalizedRect nr_0.
    nr_0.set_x_center(0.2);
    nr_0.set_y_center(0.2);
    nr_0.set_width(0.2);
    nr_0.set_height(0.2);

    // NormalizedRect nr_1.
    nr_1.set_x_center(0.4);
    nr_1.set_y_center(0.2);
    nr_1.set_width(0.2);
    nr_1.set_height(0.2);

    // NormalizedRect nr_2.
    nr_2.set_x_center(1.0);
    nr_2.set_y_center(0.3);
    nr_2.set_width(0.2);
    nr_2.set_height(0.2);

    // NormalizedRect nr_3.
    nr_3.set_x_center(0.35);
    nr_3.set_y_center(0.15);
    nr_3.set_width(0.3);
    nr_3.set_height(0.3);

    // NormalizedRect nr_4.
    nr_4.set_x_center(1.1);
    nr_4.set_y_center(0.3);
    nr_4.set_width(0.2);
    nr_4.set_height(0.2);

    // NormalizedRect nr_5.
    nr_5.set_x_center(0.5);
    nr_5.set_y_center(0.05);
    nr_5.set_width(0.2);
    nr_5.set_height(0.3);
  }
  mediapipe::NormalizedRect nr_0, nr_1, nr_2, nr_3, nr_4, nr_5;
};

TEST_F(RectangleUtilTest, OverlappingWithListLargeThreshold) {
  constexpr float kMinSimilarityThreshold = 0.15;

  std::vector<NormalizedRect> existing_rects;
  existing_rects.push_back(nr_0);
  existing_rects.push_back(nr_5);
  existing_rects.push_back(nr_2);

  EXPECT_THAT(DoesRectOverlap(nr_3, existing_rects, kMinSimilarityThreshold),
              IsOkAndHolds(true));
  EXPECT_THAT(DoesRectOverlap(nr_4, existing_rects, kMinSimilarityThreshold),
              IsOkAndHolds(true));
  EXPECT_THAT(DoesRectOverlap(nr_1, existing_rects, kMinSimilarityThreshold),
              IsOkAndHolds(false));
}

TEST_F(RectangleUtilTest, OverlappingWithListSmallThreshold) {
  constexpr float kMinSimilarityThreshold = 0.1;

  std::vector<NormalizedRect> existing_rects;
  existing_rects.push_back(nr_0);
  existing_rects.push_back(nr_5);
  existing_rects.push_back(nr_2);

  EXPECT_THAT(DoesRectOverlap(nr_3, existing_rects, kMinSimilarityThreshold),
              IsOkAndHolds(true));
  EXPECT_THAT(DoesRectOverlap(nr_4, existing_rects, kMinSimilarityThreshold),
              IsOkAndHolds(true));
  EXPECT_THAT(DoesRectOverlap(nr_1, existing_rects, kMinSimilarityThreshold),
              IsOkAndHolds(true));
}

TEST_F(RectangleUtilTest, NonOverlappingWithList) {
  constexpr float kMinSimilarityThreshold = 0.1;

  std::vector<NormalizedRect> existing_rects;
  existing_rects.push_back(nr_0);
  existing_rects.push_back(nr_3);
  existing_rects.push_back(nr_5);

  EXPECT_THAT(DoesRectOverlap(nr_2, existing_rects, kMinSimilarityThreshold),
              IsOkAndHolds(false));
  EXPECT_THAT(DoesRectOverlap(nr_4, existing_rects, kMinSimilarityThreshold),
              IsOkAndHolds(false));
}

TEST_F(RectangleUtilTest, OverlappingWithEmptyList) {
  constexpr float kMinSimilarityThreshold = 0.1;
  std::vector<NormalizedRect> existing_rects;

  EXPECT_THAT(DoesRectOverlap(nr_2, existing_rects, kMinSimilarityThreshold),
              IsOkAndHolds(false));
  EXPECT_THAT(DoesRectOverlap(nr_4, existing_rects, kMinSimilarityThreshold),
              IsOkAndHolds(false));
}

TEST_F(RectangleUtilTest, OverlapSimilarityOverlapping) {
  constexpr float kMaxAbsoluteError = 1e-4;
  constexpr float kExpectedIou = 4.0 / 9.0;
  auto rect_1 = ToRectangle(nr_1);
  auto rect_3 = ToRectangle(nr_3);
  MP_ASSERT_OK(rect_1);
  MP_ASSERT_OK(rect_3);
  EXPECT_THAT(CalculateIou(*rect_1, *rect_3),
              FloatNear(kExpectedIou, kMaxAbsoluteError));
}

TEST_F(RectangleUtilTest, OverlapSimilarityNotOverlapping) {
  constexpr float kMaxAbsoluteError = 1e-4;
  constexpr float kExpectedIou = 0.0;
  auto rect_1 = ToRectangle(nr_1);
  auto rect_2 = ToRectangle(nr_2);
  MP_ASSERT_OK(rect_1);
  MP_ASSERT_OK(rect_2);
  EXPECT_THAT(CalculateIou(*rect_1, *rect_2),
              FloatNear(kExpectedIou, kMaxAbsoluteError));
}

TEST_F(RectangleUtilTest, NormRectToRectangleSuccess) {
  const Rectangle_f kExpectedRect(/*xmin=*/0.1, /*ymin=*/0.1,
                                  /*width=*/0.2, /*height=*/0.2);
  EXPECT_THAT(ToRectangle(nr_0), IsOkAndHolds(kExpectedRect));
}

TEST_F(RectangleUtilTest, NormRectToRectangleFail) {
  mediapipe::NormalizedRect invalid_nr;
  invalid_nr.set_x_center(0.2);
  EXPECT_THAT(ToRectangle(invalid_nr), testing::Not(IsOk()));

  invalid_nr.set_y_center(0.2);
  invalid_nr.set_width(-0.2);
  invalid_nr.set_height(0.2);
  EXPECT_THAT(ToRectangle(invalid_nr), testing::Not(IsOk()));

  invalid_nr.set_width(0.2);
  invalid_nr.set_height(-0.2);
  EXPECT_THAT(ToRectangle(invalid_nr), testing::Not(IsOk()));
}

}  // namespace
}  // namespace mediapipe
