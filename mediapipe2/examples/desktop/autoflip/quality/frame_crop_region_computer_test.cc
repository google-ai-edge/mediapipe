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

#include "mediapipe/examples/desktop/autoflip/quality/frame_crop_region_computer.h"

#include <memory>

#include "absl/memory/memory.h"
#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/cropping.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace autoflip {

using ::testing::HasSubstr;

const int kSegmentMaxLength = 10;
const int kTargetWidth = 500;
const int kTargetHeight = 1000;

// Makes a rectangle given the corner (x, y) and the size (width, height).
Rect MakeRect(const int x, const int y, const int width, const int height) {
  Rect rect;
  rect.set_x(x);
  rect.set_y(y);
  rect.set_width(width);
  rect.set_height(height);
  return rect;
}

// Adds a detection to the key frame info given its location, whether it is
// required, and its score. The score is default to 1.0.
void AddDetection(const Rect& rect, const bool is_required,
                  KeyFrameInfo* key_frame_info, const float score = 1.0) {
  auto* detection = key_frame_info->mutable_detections()->add_detections();
  *(detection->mutable_location()) = rect;
  detection->set_score(score);
  detection->set_is_required(is_required);
}

// Makes key frame crop options given target width and height.
KeyFrameCropOptions MakeKeyFrameCropOptions(const int target_width,
                                            const int target_height) {
  KeyFrameCropOptions options;
  options.set_target_width(target_width);
  options.set_target_height(target_height);
  return options;
}

// Checks whether rectangle a is inside rectangle b.
bool CheckRectIsInside(const Rect& rect_a, const Rect& rect_b) {
  return (rect_b.x() <= rect_a.x() && rect_b.y() <= rect_a.y() &&
          rect_b.x() + rect_b.width() >= rect_a.x() + rect_a.width() &&
          rect_b.y() + rect_b.height() >= rect_a.y() + rect_a.height());
}

// Checks whether two rectangles are equal.
bool CheckRectsEqual(const Rect& rect1, const Rect& rect2) {
  return (rect1.x() == rect2.x() && rect1.y() == rect2.y() &&
          rect1.width() == rect2.width() && rect1.height() == rect2.height());
}

// Checks whether two rectangles have non-zero overlapping area.
bool CheckRectsOverlap(const Rect& rect1, const Rect& rect2) {
  const int x1_left = rect1.x(), x1_right = rect1.x() + rect1.width();
  const int y1_top = rect1.y(), y1_bottom = rect1.y() + rect1.height();
  const int x2_left = rect2.x(), x2_right = rect2.x() + rect2.width();
  const int y2_top = rect2.y(), y2_bottom = rect2.y() + rect2.height();
  const int x_left = std::max(x1_left, x2_left);
  const int x_right = std::min(x1_right, x2_right);
  const int y_top = std::max(y1_top, y2_top);
  const int y_bottom = std::min(y1_bottom, y2_bottom);
  return (x_right > x_left && y_bottom > y_top);
}

// Checks that all the required regions in the detections in KeyFrameInfo are
// covered in the KeyFrameCropResult.
void CheckRequiredRegionsAreCovered(const KeyFrameInfo& key_frame_info,
                                    const KeyFrameCropResult& result) {
  bool has_required = false;
  for (int i = 0; i < key_frame_info.detections().detections_size(); ++i) {
    const auto& detection = key_frame_info.detections().detections(i);
    if (detection.is_required()) {
      has_required = true;
      EXPECT_TRUE(
          CheckRectIsInside(detection.location(), result.required_region()));
    }
  }
  EXPECT_EQ(has_required, !result.required_region_is_empty());
  if (has_required) {
    EXPECT_FALSE(result.region_is_empty());
    EXPECT_TRUE(CheckRectIsInside(result.required_region(), result.region()));
  }
}

// Testable class that can access protected types and methods in the class.
class TestableFrameCropRegionComputer : public FrameCropRegionComputer {
 public:
  explicit TestableFrameCropRegionComputer(const KeyFrameCropOptions& options)
      : FrameCropRegionComputer(options) {}
  using FrameCropRegionComputer::CoverType;
  using FrameCropRegionComputer::ExpandRectUnderConstraints;
  using FrameCropRegionComputer::ExpandSegmentUnderConstraint;
  using FrameCropRegionComputer::FULLY_COVERED;
  using FrameCropRegionComputer::LeftPoint;  // int
  using FrameCropRegionComputer::NOT_COVERED;
  using FrameCropRegionComputer::PARTIALLY_COVERED;
  using FrameCropRegionComputer::RightPoint;  // int
  using FrameCropRegionComputer::Segment;     // std::pair<int, int>
  using FrameCropRegionComputer::UpdateCropRegionScore;

  // Makes a segment from two endpoints.
  static Segment MakeSegment(const LeftPoint left, const RightPoint right) {
    return std::make_pair(left, right);
  }

  // Checks that two segments are equal.
  static bool CheckSegmentsEqual(const Segment& segment1,
                                 const Segment& segment2) {
    return (segment1.first == segment2.first &&
            segment1.second == segment2.second);
  }
};
using TestClass = TestableFrameCropRegionComputer;

// Returns an instance of the testable class given
// non_required_region_min_coverage_fraction.
std::unique_ptr<TestClass> GetTestableClass(
    const float non_required_region_min_coverage_fraction = 0.5) {
  KeyFrameCropOptions options;
  options.set_non_required_region_min_coverage_fraction(
      non_required_region_min_coverage_fraction);
  auto test_class = absl::make_unique<TestClass>(options);
  return test_class;
}

// Checks that ExpandSegmentUnderConstraint checks output pointers are not null.
TEST(FrameCropRegionComputerTest, ExpandSegmentUnderConstraintCheckNull) {
  auto test_class = GetTestableClass();
  TestClass::CoverType cover_type;
  TestClass::Segment base_segment = TestClass::MakeSegment(10, 15);
  TestClass::Segment segment_to_add = TestClass::MakeSegment(5, 8);
  TestClass::Segment combined_segment;
  // Combined segment is null.
  auto status = test_class->ExpandSegmentUnderConstraint(
      segment_to_add, base_segment, kSegmentMaxLength, nullptr, &cover_type);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Combined segment is null."));
  // Cover type is null.
  status = test_class->ExpandSegmentUnderConstraint(
      segment_to_add, base_segment, kSegmentMaxLength, &combined_segment,
      nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Cover type is null."));
}

// Checks that ExpandSegmentUnderConstraint checks input segments are valid.
TEST(FrameCropRegionComputerTest, ExpandSegmentUnderConstraintCheckValid) {
  auto test_class = GetTestableClass();
  TestClass::CoverType cover_type;
  TestClass::Segment combined_segment;

  // Invalid base segment.
  TestClass::Segment base_segment = TestClass::MakeSegment(15, 10);
  TestClass::Segment segment_to_add = TestClass::MakeSegment(5, 8);
  auto status = test_class->ExpandSegmentUnderConstraint(
      segment_to_add, base_segment, kSegmentMaxLength, &combined_segment,
      &cover_type);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Invalid base segment."));

  // Invalid segment to add.
  base_segment = TestClass::MakeSegment(10, 15);
  segment_to_add = TestClass::MakeSegment(8, 5);
  status = test_class->ExpandSegmentUnderConstraint(
      segment_to_add, base_segment, kSegmentMaxLength, &combined_segment,
      &cover_type);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Invalid segment to add."));

  // Base segment exceeds max length.
  base_segment = TestClass::MakeSegment(10, 100);
  segment_to_add = TestClass::MakeSegment(5, 8);
  status = test_class->ExpandSegmentUnderConstraint(
      segment_to_add, base_segment, kSegmentMaxLength, &combined_segment,
      &cover_type);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              HasSubstr("Base segment length exceeds max length."));
}

// Checks that ExpandSegmentUnderConstraint handles case 1 properly: the length
// of the union of the two segments is not larger than the maximum length.
TEST(FrameCropRegionComputerTest, ExpandSegmentUnderConstraintCase1) {
  auto test_class = GetTestableClass();
  TestClass::Segment combined_segment;
  TestClass::CoverType cover_type;
  TestClass::Segment base_segment = TestClass::MakeSegment(5, 10);
  TestClass::Segment segment_to_add = TestClass::MakeSegment(3, 8);
  MP_EXPECT_OK(test_class->ExpandSegmentUnderConstraint(
      segment_to_add, base_segment, kSegmentMaxLength, &combined_segment,
      &cover_type));
  EXPECT_EQ(cover_type, TestClass::FULLY_COVERED);
  EXPECT_TRUE(TestClass::CheckSegmentsEqual(combined_segment,
                                            TestClass::MakeSegment(3, 10)));
}

// Checks that ExpandSegmentUnderConstraint handles case 2 properly: the union
// of the two segments exceeds the maximum length, but the union of the base
// segment with the minimum coverage fraction of the new segment is within the
// maximum length.
TEST(FrameCropRegionComputerTest, ExpandSegmentUnderConstraintCase2) {
  TestClass::Segment combined_segment;
  TestClass::CoverType cover_type;
  TestClass::Segment base_segment = TestClass::MakeSegment(4, 8);
  TestClass::Segment segment_to_add = TestClass::MakeSegment(0, 16);
  auto test_class = GetTestableClass();
  MP_EXPECT_OK(test_class->ExpandSegmentUnderConstraint(
      segment_to_add, base_segment, kSegmentMaxLength, &combined_segment,
      &cover_type));
  EXPECT_EQ(cover_type, TestClass::PARTIALLY_COVERED);
  EXPECT_TRUE(TestClass::CheckSegmentsEqual(combined_segment,
                                            TestClass::MakeSegment(4, 12)));
}

// Checks that ExpandSegmentUnderConstraint handles case 3 properly: the union
// of the base segment with the minimum coverage fraction of the new segment
// exceeds the maximum length.
TEST(FrameCropRegionComputerTest, ExpandSegmentUnderConstraintCase3) {
  TestClass::Segment combined_segment;
  TestClass::CoverType cover_type;
  auto test_class = GetTestableClass();
  TestClass::Segment base_segment = TestClass::MakeSegment(6, 14);
  TestClass::Segment segment_to_add = TestClass::MakeSegment(0, 4);
  MP_EXPECT_OK(test_class->ExpandSegmentUnderConstraint(
      segment_to_add, base_segment, kSegmentMaxLength, &combined_segment,
      &cover_type));
  EXPECT_EQ(cover_type, TestClass::NOT_COVERED);
  EXPECT_TRUE(TestClass::CheckSegmentsEqual(combined_segment, base_segment));
}

// Checks that ExpandRectUnderConstraints checks output pointers are not null.
TEST(FrameCropRegionComputerTest, ExpandRectUnderConstraintsChecksNotNull) {
  auto test_class = GetTestableClass();
  TestClass::CoverType cover_type;
  Rect base_rect, rect_to_add;
  // Base rect is null.
  auto status = test_class->ExpandRectUnderConstraints(
      rect_to_add, kTargetWidth, kTargetHeight, nullptr, &cover_type);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Base rect is null."));
  // Cover type is null.
  status = test_class->ExpandRectUnderConstraints(
      rect_to_add, kTargetWidth, kTargetHeight, &base_rect, nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Cover type is null."));
}

// Checks that ExpandRectUnderConstraints checks base rect is valid.
TEST(FrameCropRegionComputerTest, ExpandRectUnderConstraintsChecksBaseValid) {
  auto test_class = GetTestableClass();
  TestClass::CoverType cover_type;
  Rect base_rect = MakeRect(0, 0, 2 * kTargetWidth, 2 * kTargetHeight);
  Rect rect_to_add;
  const auto status = test_class->ExpandRectUnderConstraints(
      rect_to_add, kTargetWidth, kTargetHeight, &base_rect, &cover_type);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              HasSubstr("Base rect already exceeds target size."));
}

// Checks that ExpandRectUnderConstraints properly handles the case where the
// rectangle to be added can be fully covered.
TEST(FrameCropRegionComputerTest, ExpandRectUnderConstraintsFullyCovered) {
  auto test_class = GetTestableClass();
  TestClass::CoverType cover_type;
  Rect base_rect = MakeRect(0, 0, 50, 50);
  Rect rect_to_add = MakeRect(30, 30, 30, 30);
  MP_EXPECT_OK(test_class->ExpandRectUnderConstraints(
      rect_to_add, kTargetWidth, kTargetHeight, &base_rect, &cover_type));
  EXPECT_EQ(cover_type, TestClass::FULLY_COVERED);
  EXPECT_TRUE(CheckRectsEqual(base_rect, MakeRect(0, 0, 60, 60)));
}

// Checks that ExpandRectUnderConstraints properly handles the case where the
// rectangle to be added can be partially covered.
TEST(FrameCropRegionComputerTest, ExpandRectUnderConstraintsPartiallyCovered) {
  auto test_class = GetTestableClass();
  TestClass::CoverType cover_type;
  // Rectangle to be added can be partially covered in both both dimensions.
  Rect base_rect = MakeRect(0, 0, 500, 500);
  Rect rect_to_add = MakeRect(0, 300, 600, 900);
  MP_EXPECT_OK(test_class->ExpandRectUnderConstraints(
      rect_to_add, kTargetWidth, kTargetHeight, &base_rect, &cover_type));
  EXPECT_EQ(cover_type, TestClass::PARTIALLY_COVERED);
  EXPECT_TRUE(CheckRectsEqual(base_rect, MakeRect(0, 0, 500, 975)));

  // Rectangle to be added can be fully covered in one dimension and partially
  // covered in the other dimension.
  base_rect = MakeRect(0, 0, 400, 500);
  rect_to_add = MakeRect(100, 300, 400, 900);
  MP_EXPECT_OK(test_class->ExpandRectUnderConstraints(
      rect_to_add, kTargetWidth, kTargetHeight, &base_rect, &cover_type));
  EXPECT_EQ(cover_type, TestClass::PARTIALLY_COVERED);
  EXPECT_TRUE(CheckRectsEqual(base_rect, MakeRect(0, 0, 500, 975)));
}

// Checks that ExpandRectUnderConstraints properly handles the case where the
// rectangle to be added cannot be covered.
TEST(FrameCropRegionComputerTest, ExpandRectUnderConstraintsNotCovered) {
  TestClass::CoverType cover_type;
  auto test_class = GetTestableClass();
  Rect base_rect = MakeRect(0, 0, 500, 500);
  Rect rect_to_add = MakeRect(550, 300, 100, 900);
  MP_EXPECT_OK(test_class->ExpandRectUnderConstraints(
      rect_to_add, kTargetWidth, kTargetHeight, &base_rect, &cover_type));
  EXPECT_EQ(cover_type, TestClass::NOT_COVERED);  // no overlap in x dimension
  EXPECT_TRUE(CheckRectsEqual(base_rect, MakeRect(0, 0, 500, 500)));
}

// Checks that ComputeFrameCropRegion handles the case of empty detections.
TEST(FrameCropRegionComputerTest, HandlesEmptyDetections) {
  const auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  EXPECT_TRUE(crop_result.region_is_empty());
}

// Checks that ComputeFrameCropRegion covers required regions when their union
// is within target size.
TEST(FrameCropRegionComputerTest, CoversRequiredWithinTargetSize) {
  const auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(100, 100, 100, 200), true, &key_frame_info);
  AddDetection(MakeRect(200, 400, 300, 500), true, &key_frame_info);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  CheckRequiredRegionsAreCovered(key_frame_info, crop_result);
  EXPECT_TRUE(CheckRectsEqual(MakeRect(100, 100, 400, 800),
                              crop_result.required_region()));
  EXPECT_TRUE(
      CheckRectsEqual(crop_result.region(), crop_result.required_region()));
  EXPECT_TRUE(crop_result.are_required_regions_covered_in_target_size());
}

// Checks that ComputeFrameCropRegion covers required regions when their union
// exceeds target size.
TEST(FrameCropRegionComputerTest, CoversRequiredExceedingTargetSize) {
  const auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(0, 0, 100, 500), true, &key_frame_info);
  AddDetection(MakeRect(200, 400, 500, 500), true, &key_frame_info);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  CheckRequiredRegionsAreCovered(key_frame_info, crop_result);
  EXPECT_TRUE(CheckRectsEqual(MakeRect(0, 0, 700, 900), crop_result.region()));
  EXPECT_TRUE(
      CheckRectsEqual(crop_result.region(), crop_result.required_region()));
  EXPECT_FALSE(crop_result.are_required_regions_covered_in_target_size());
}

// Checks that ComputeFrameCropRegion handles the case of only non-required
// regions and the region fits in the target size.
TEST(FrameCropRegionComputerTest,
     HandlesOnlyNonRequiedRegionsInsideTargetSize) {
  const auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(300, 600, 100, 100), false, &key_frame_info);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  EXPECT_TRUE(crop_result.required_region_is_empty());
  EXPECT_FALSE(crop_result.region_is_empty());
  EXPECT_TRUE(
      CheckRectsEqual(key_frame_info.detections().detections(0).location(),
                      crop_result.region()));
}

// Checks that ComputeFrameCropRegion handles the case of only non-required
// regions and the region exceeds the target size.
TEST(FrameCropRegionComputerTest,
     HandlesOnlyNonRequiedRegionsExceedingTargetSize) {
  const auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(300, 600, 700, 100), false, &key_frame_info);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  EXPECT_TRUE(crop_result.required_region_is_empty());
  EXPECT_FALSE(crop_result.region_is_empty());
  EXPECT_TRUE(
      CheckRectsEqual(MakeRect(475, 600, 350, 100), crop_result.region()));
  EXPECT_EQ(crop_result.fraction_non_required_covered(), 0.0);
  EXPECT_TRUE(
      CheckRectIsInside(crop_result.region(),
                        key_frame_info.detections().detections(0).location()));
}

// Checks that ComputeFrameCropRegion covers non-required regions when their
// union fits within target size.
TEST(FrameCropRegionComputerTest, CoversNonRequiredInsideTargetSize) {
  const auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(0, 0, 100, 500), true, &key_frame_info);
  AddDetection(MakeRect(300, 600, 100, 100), false, &key_frame_info);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  CheckRequiredRegionsAreCovered(key_frame_info, crop_result);
  EXPECT_TRUE(CheckRectsEqual(MakeRect(0, 0, 400, 700), crop_result.region()));
  EXPECT_TRUE(crop_result.are_required_regions_covered_in_target_size());
  EXPECT_EQ(crop_result.fraction_non_required_covered(), 1.0);
  for (int i = 0; i < key_frame_info.detections().detections_size(); ++i) {
    EXPECT_TRUE(
        CheckRectIsInside(key_frame_info.detections().detections(i).location(),
                          crop_result.region()));
  }
}

// Checks that ComputeFrameCropRegion does not cover non-required regions that
// are outside the target size.
TEST(FrameCropRegionComputerTest, DoesNotCoverNonRequiredExceedingTargetSize) {
  const auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(0, 0, 500, 1000), true, &key_frame_info);
  AddDetection(MakeRect(500, 0, 100, 100), false, &key_frame_info);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  CheckRequiredRegionsAreCovered(key_frame_info, crop_result);
  EXPECT_TRUE(CheckRectsEqual(MakeRect(0, 0, 500, 1000), crop_result.region()));
  EXPECT_TRUE(crop_result.are_required_regions_covered_in_target_size());
  EXPECT_EQ(crop_result.fraction_non_required_covered(), 0.0);
  EXPECT_FALSE(
      CheckRectIsInside(key_frame_info.detections().detections(1).location(),
                        crop_result.region()));
}

// Checks that ComputeFrameCropRegion partially covers non-required regions that
// can partially fit in the target size.
TEST(FrameCropRegionComputerTest,
     PartiallyCoversNonRequiredContainingTargetSize) {
  const auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(100, 0, 350, 1000), true, &key_frame_info);
  AddDetection(MakeRect(0, 0, 650, 100), false, &key_frame_info);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  CheckRequiredRegionsAreCovered(key_frame_info, crop_result);
  EXPECT_TRUE(
      CheckRectsEqual(MakeRect(100, 0, 387, 1000), crop_result.region()));
  EXPECT_TRUE(crop_result.are_required_regions_covered_in_target_size());
  EXPECT_EQ(crop_result.fraction_non_required_covered(), 0.0);
  EXPECT_TRUE(
      CheckRectsOverlap(key_frame_info.detections().detections(1).location(),
                        crop_result.region()));
}

// Checks that ComputeFrameCropRegion covers non-required regions when the
// required regions exceed target size.
TEST(FrameCropRegionComputerTest,
     CoversNonRequiredWhenRequiredExceedsTargetSize) {
  const auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(0, 0, 600, 1000), true, &key_frame_info);
  AddDetection(MakeRect(450, 0, 100, 100), false, &key_frame_info);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  CheckRequiredRegionsAreCovered(key_frame_info, crop_result);
  EXPECT_TRUE(CheckRectsEqual(MakeRect(0, 0, 600, 1000), crop_result.region()));
  EXPECT_FALSE(crop_result.are_required_regions_covered_in_target_size());
  EXPECT_EQ(crop_result.fraction_non_required_covered(), 1.0);
  for (int i = 0; i < key_frame_info.detections().detections_size(); ++i) {
    EXPECT_TRUE(
        CheckRectIsInside(key_frame_info.detections().detections(i).location(),
                          crop_result.region()));
  }
}

// Checks that ComputeFrameCropRegion does not extend the crop region when
// the non-required region is too far.
TEST(FrameCropRegionComputerTest,
     DoesNotExtendRegionWhenNonRequiredRegionIsTooFar) {
  const auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(0, 0, 400, 400), true, &key_frame_info);
  AddDetection(MakeRect(600, 0, 100, 100), false, &key_frame_info);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  CheckRequiredRegionsAreCovered(key_frame_info, crop_result);
  EXPECT_TRUE(CheckRectsEqual(MakeRect(0, 0, 400, 400), crop_result.region()));
  EXPECT_TRUE(crop_result.are_required_regions_covered_in_target_size());
  EXPECT_EQ(crop_result.fraction_non_required_covered(), 0.0);
  EXPECT_FALSE(
      CheckRectsOverlap(key_frame_info.detections().detections(1).location(),
                        crop_result.region()));
}

// Checks that ComputeFrameCropRegion computes the score correctly when the
// aggregation type is maximum.
TEST(FrameCropRegionComputerTest, ComputesScoreWhenAggregationIsMaximum) {
  auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  options.set_score_aggregation_type(KeyFrameCropOptions::MAXIMUM);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(0, 0, 400, 400), true, &key_frame_info, 0.1);
  AddDetection(MakeRect(300, 300, 200, 500), true, &key_frame_info, 0.9);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  EXPECT_FLOAT_EQ(crop_result.region_score(), 0.9f);
}

// Checks that ComputeFrameCropRegion computes the score correctly when the
// aggregation type is sum required regions.
TEST(FrameCropRegionComputerTest, ComputesScoreWhenAggregationIsSumRequired) {
  auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  options.set_score_aggregation_type(KeyFrameCropOptions::SUM_REQUIRED);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(0, 0, 400, 400), true, &key_frame_info, 0.1);
  AddDetection(MakeRect(300, 300, 200, 500), true, &key_frame_info, 0.9);
  AddDetection(MakeRect(300, 300, 200, 500), false, &key_frame_info, 0.5);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  EXPECT_FLOAT_EQ(crop_result.region_score(), 1.0f);
}

// Checks that ComputeFrameCropRegion computes the score correctly when the
// aggregation type is sum all covered regions.
TEST(FrameCropRegionComputerTest, ComputesScoreWhenAggregationIsSumAll) {
  auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  options.set_score_aggregation_type(KeyFrameCropOptions::SUM_ALL);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(0, 0, 400, 400), true, &key_frame_info, 0.1);
  AddDetection(MakeRect(300, 300, 200, 500), true, &key_frame_info, 0.9);
  AddDetection(MakeRect(300, 300, 200, 500), false, &key_frame_info, 0.5);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  EXPECT_FLOAT_EQ(crop_result.region_score(), 1.5f);
}

// Checks that ComputeFrameCropRegion computes the score correctly when the
// aggregation type is constant.
TEST(FrameCropRegionComputerTest, ComputesScoreWhenAggregationIsConstant) {
  auto options = MakeKeyFrameCropOptions(kTargetWidth, kTargetHeight);
  options.set_score_aggregation_type(KeyFrameCropOptions::CONSTANT);
  FrameCropRegionComputer computer(options);
  KeyFrameInfo key_frame_info;
  AddDetection(MakeRect(0, 0, 400, 400), true, &key_frame_info, 0.1);
  AddDetection(MakeRect(300, 300, 200, 500), true, &key_frame_info, 0.9);
  AddDetection(MakeRect(300, 300, 200, 500), false, &key_frame_info, 0.5);
  KeyFrameCropResult crop_result;
  MP_EXPECT_OK(computer.ComputeFrameCropRegion(key_frame_info, &crop_result));
  EXPECT_FLOAT_EQ(crop_result.region_score(), 1.0f);
}
}  // namespace autoflip
}  // namespace mediapipe
