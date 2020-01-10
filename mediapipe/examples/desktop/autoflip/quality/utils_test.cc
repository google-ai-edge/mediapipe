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

#include "mediapipe/examples/desktop/autoflip/quality/utils.h"

#include <random>
#include <vector>

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/cropping.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace autoflip {
namespace {

using ::testing::HasSubstr;

const int64 kTimestamp = 0;
const int kOriginalWidth = 100;
const int kOriginalHeight = 100;
const double kTargetAspectRatio = 1.5;
const int kNumKeyFrames = 5;
const int64 kKeyFrameTimestampDiff = 1e6 / kNumKeyFrames;
const int kTargetWidth = 50;
const int kTargetHeight = 50;

// Makes a rectangle given the corner (x, y) and the size (width, height).
Rect MakeRect(const int x, const int y, const int width, const int height) {
  Rect rect;
  rect.set_x(x);
  rect.set_y(y);
  rect.set_width(width);
  rect.set_height(height);
  return rect;
}

// Makes a rectangle given the center (center_x, center_y) and the half size
// (half_width, half_height).
Rect MakeRectFromCenterAndHalfSize(const float center_x, const float center_y,
                                   const float half_width,
                                   const float half_height) {
  Rect rect;
  rect.set_x(center_x - half_width);
  rect.set_y(center_y - half_height);
  rect.set_width(half_width * 2);
  rect.set_height(half_height * 2);
  return rect;
}

// Computes the center of a rectangle as a pair (center_x, center_y).
std::pair<float, float> RectCenter(const Rect& rect) {
  return std::make_pair(rect.x() + rect.width() / 2.0,
                        rect.y() + rect.height() / 2.0);
}

// Makes a normalized rectangle given the corner (x, y) and the size (width,
// height).
RectF MakeRectF(const double x, const double y, const double width,
                const double height) {
  RectF rectf;
  rectf.set_x(x);
  rectf.set_y(y);
  rectf.set_width(width);
  rectf.set_height(height);
  return rectf;
}

// Adds a detection to the detection set given location.
void AddDetectionFromLocation(const Rect& loc, DetectionSet* detections) {
  auto* detection = detections->add_detections();
  *(detection->mutable_location()) = loc;
}

// Adds a detection to the detection set given normalized location.
void AddDetectionFromNormalizedLocation(const RectF& normalized_loc,
                                        DetectionSet* detections) {
  auto* detection = detections->add_detections();
  *(detection->mutable_location_normalized()) = normalized_loc;
}

// Checks whether two rectangles are the same, with an optional scaling factor.
bool CheckRectsEqual(const Rect& rect1, const Rect& rect2,
                     const double scale_x = 1.0, const double scale_y = 1.0) {
  return (static_cast<int>(round(scale_x * rect2.x())) == rect1.x() &&
          static_cast<int>(round(scale_y * rect2.y())) == rect1.y() &&
          static_cast<int>(round(scale_x * rect2.width())) == rect1.width() &&
          static_cast<int>(round(scale_y * rect2.height())) == rect1.height());
}

// Adds a detection to the detection set given its score and whether it is
// required.
void AddDetectionFromScoreAndIsRequired(const double score,
                                        const bool is_required,
                                        DetectionSet* detections) {
  auto* detection = detections->add_detections();
  detection->set_score(score);
  detection->set_is_required(is_required);
}

// Returns default values for KeyFrameInfos. Populates timestamps using the
// default spacing kKeyFrameTimestampDiff starting from 0.
std::vector<KeyFrameInfo> GetDefaultKeyFrameInfos() {
  std::vector<KeyFrameInfo> key_frame_infos(kNumKeyFrames);
  for (int i = 0; i < kNumKeyFrames; ++i) {
    key_frame_infos[i].set_timestamp_ms(kKeyFrameTimestampDiff * i);
  }
  return key_frame_infos;
}

// Returns default settings for KeyFrameCropOptions. Populates target size to be
// the default target size.
KeyFrameCropOptions GetDefaultKeyFrameCropOptions() {
  KeyFrameCropOptions key_frame_crop_options;
  key_frame_crop_options.set_target_width(kTargetWidth);
  key_frame_crop_options.set_target_height(kTargetHeight);
  return key_frame_crop_options;
}

// Returns default values for KeyFrameCropResults. Sets each frame to have
// covered all the required regions and non-required regions, and have required
// crop region (10, 10+20) x (10, 10+20), (full) crop region (0, 50) x (0, 50),
// and region score 1.0.
std::vector<KeyFrameCropResult> GetDefaultKeyFrameCropResults() {
  std::vector<KeyFrameCropResult> key_frame_crop_results(kNumKeyFrames);
  for (int i = 0; i < kNumKeyFrames; ++i) {
    key_frame_crop_results[i].set_are_required_regions_covered_in_target_size(
        true);
    key_frame_crop_results[i].set_fraction_non_required_covered(1.0);
    key_frame_crop_results[i].set_region_is_empty(false);
    key_frame_crop_results[i].set_required_region_is_empty(false);
    *(key_frame_crop_results[i].mutable_region()) = MakeRect(0, 0, 50, 50);
    *(key_frame_crop_results[i].mutable_required_region()) =
        MakeRect(10, 10, 20, 20);
    key_frame_crop_results[i].set_region_score(1.0);
  }
  return key_frame_crop_results;
}

// Checks that ScaleRect properly scales Rect and RectF objects.
TEST(UtilTest, ScaleRect) {
  Rect scaled_rect;
  ScaleRect(MakeRect(10, 10, 20, 30), 1.5, 2.0, &scaled_rect);
  EXPECT_TRUE(CheckRectsEqual(scaled_rect, MakeRect(15, 20, 30, 60)));

  ScaleRect(MakeRectF(0.5, 0.9, 1.36, 0.748), 100, 50, &scaled_rect);
  EXPECT_TRUE(CheckRectsEqual(scaled_rect, MakeRect(50, 45, 136, 37)));
}

// Checks that NormalizedRectToRect properly converts a RectF object to a Rect
// object given width and height.
TEST(UtilTest, NormalizedRectToRect) {
  const RectF normalized_rect = MakeRectF(0.1, 1.0, 2.5, 0.9);
  Rect rect;
  NormalizedRectToRect(normalized_rect, 100, 100, &rect);
  EXPECT_TRUE(CheckRectsEqual(rect, MakeRect(10, 100, 250, 90)));
}

// Checks that ClampRect properly clamps a Rect object in [x0, y0] and [x1, y1].
TEST(UtilTest, ClampRect) {
  // Overlaps at a corner.
  Rect rect = MakeRect(-10, -10, 80, 20);
  MP_EXPECT_OK(ClampRect(0, 0, 100, 100, &rect));
  EXPECT_TRUE(CheckRectsEqual(rect, MakeRect(0, 0, 70, 10)));
  // Overlaps on a side.
  rect = MakeRect(10, -10, 80, 20);
  MP_EXPECT_OK(ClampRect(0, 0, 100, 100, &rect));
  EXPECT_TRUE(CheckRectsEqual(rect, MakeRect(10, 0, 80, 10)));
  // Inside.
  rect = MakeRect(10, 10, 80, 10);
  MP_EXPECT_OK(ClampRect(0, 0, 100, 100, &rect));
  EXPECT_TRUE(CheckRectsEqual(rect, MakeRect(10, 10, 80, 10)));
  // Outside.
  rect = MakeRect(-10, 10, 0, 0);
  EXPECT_FALSE(ClampRect(0, 0, 100, 100, &rect).ok());
}

// Checks that ClampRect properly clamps a Rect object in [0, 0] and [width,
// height].
TEST(UtilTest, ClampRectConvenienceFunction) {
  Rect rect = MakeRect(-10, 0, 80, 10);
  MP_EXPECT_OK(ClampRect(kOriginalWidth, kOriginalHeight, &rect));
  EXPECT_TRUE(CheckRectsEqual(rect, MakeRect(0, 0, 70, 10)));
  rect = MakeRect(-10, 0, 120, 10);
  MP_EXPECT_OK(ClampRect(kOriginalWidth, kOriginalHeight, &rect));
  EXPECT_TRUE(CheckRectsEqual(rect, MakeRect(0, 0, 100, 10)));
  rect = MakeRect(10, 0, 70, 120);
  MP_EXPECT_OK(ClampRect(kOriginalWidth, kOriginalHeight, &rect));
  EXPECT_TRUE(CheckRectsEqual(rect, MakeRect(10, 0, 70, 100)));
}

// Checks that RectUnion properly takes the union of two Rect objects.
TEST(UtilTest, RectUnion) {
  // Base rectangle and new rectangle are partially overlapping.
  Rect base_rect = MakeRect(40, 40, 40, 40);
  RectUnion(MakeRect(20, 30, 40, 40), &base_rect);
  EXPECT_TRUE(CheckRectsEqual(base_rect, MakeRect(20, 30, 60, 50)));
  // Base rectangle contains new rectangle.
  base_rect = MakeRect(40, 40, 40, 40);
  RectUnion(MakeRect(50, 50, 10, 10), &base_rect);
  EXPECT_TRUE(CheckRectsEqual(base_rect, MakeRect(40, 40, 40, 40)));
  // Base rectangle is contained by new rectangle.
  base_rect = MakeRect(40, 40, 40, 40);
  RectUnion(MakeRect(30, 30, 50, 50), &base_rect);
  EXPECT_TRUE(CheckRectsEqual(base_rect, MakeRect(30, 30, 50, 50)));
  // Base rectangle and new rectangle are disjoint.
  base_rect = MakeRect(40, 40, 40, 40);
  RectUnion(MakeRect(15, 25, 20, 5), &base_rect);
  EXPECT_TRUE(CheckRectsEqual(base_rect, MakeRect(15, 25, 65, 55)));
}

// Checks that PackCropFrame fails on nullptr return object.
TEST(UtilTest, PackKeyFrameInfoFailsOnNullObject) {
  DetectionSet detections;
  const int feature_width = kOriginalWidth, feature_height = kOriginalHeight;

  KeyFrameInfo* key_frame_info_ptr = nullptr;
  const auto status =
      PackKeyFrameInfo(kTimestamp, detections, kOriginalWidth, kOriginalHeight,
                       feature_width, feature_height, key_frame_info_ptr);
  EXPECT_FALSE(status.ok());
}

// Checks that PackCropFrame fails on invalid frame size.
TEST(UtilTest, PackKeyFrameInfoFailsOnInvalidFrameSize) {
  DetectionSet detections;
  const int feature_width = -1, feature_height = 0;

  KeyFrameInfo key_frame_info;
  const auto status =
      PackKeyFrameInfo(kTimestamp, detections, kOriginalWidth, kOriginalHeight,
                       feature_width, feature_height, &key_frame_info)
          .ToString();
  EXPECT_THAT(status, testing::HasSubstr("Invalid frame size"));
}

// Checks that PackCropFrame correctly packs timestamp.
TEST(UtilTest, PackKeyFrameInfoPacksTimestamp) {
  DetectionSet detections;
  const int feature_width = kOriginalWidth, feature_height = kOriginalHeight;

  KeyFrameInfo key_frame_info;
  const auto status =
      PackKeyFrameInfo(kTimestamp, detections, kOriginalWidth, kOriginalHeight,
                       feature_width, feature_height, &key_frame_info);

  MP_EXPECT_OK(status);
  EXPECT_EQ(key_frame_info.timestamp_ms(), kTimestamp);
}

// Checks that PackCropFrame correctly packs detections.
TEST(UtilTest, PackKeyFrameInfoPacksDetections) {
  DetectionSet detections;
  AddDetectionFromLocation(MakeRect(0, 0, 10, 10), &detections);
  AddDetectionFromLocation(MakeRect(20, 20, 30, 10), &detections);
  const int feature_width = kOriginalWidth, feature_height = kOriginalHeight;

  KeyFrameInfo key_frame_info;
  const auto status =
      PackKeyFrameInfo(kTimestamp, detections, kOriginalWidth, kOriginalHeight,
                       feature_width, feature_height, &key_frame_info);

  MP_EXPECT_OK(status);
  EXPECT_EQ(key_frame_info.detections().detections_size(),
            detections.detections_size());
  for (int i = 0; i < detections.detections_size(); ++i) {
    const auto& original_rect = detections.detections(i).location();
    const auto& rect = key_frame_info.detections().detections(i).location();
    EXPECT_TRUE(CheckRectsEqual(rect, original_rect));
  }
}

// Checks that PackCropFrame correctly converts normalized location to location.
TEST(UtilTest, PackKeyFrameInfoUnnormalizesLocations) {
  DetectionSet detections;
  AddDetectionFromNormalizedLocation(MakeRectF(0.1, 0.1, 0.1, 0.1),
                                     &detections);
  AddDetectionFromNormalizedLocation(MakeRectF(0.242, 0.256, 0.378, 0.399),
                                     &detections);
  const int feature_width = kOriginalWidth, feature_height = kOriginalHeight;

  KeyFrameInfo key_frame_info;
  const auto status =
      PackKeyFrameInfo(kTimestamp, detections, kOriginalWidth, kOriginalHeight,
                       feature_width, feature_height, &key_frame_info);

  MP_EXPECT_OK(status);
  const auto& out_rect1 = key_frame_info.detections().detections(0).location();
  const auto& out_rect2 = key_frame_info.detections().detections(1).location();
  EXPECT_TRUE(CheckRectsEqual(out_rect1, MakeRect(10, 10, 10, 10)));
  EXPECT_TRUE(CheckRectsEqual(out_rect2, MakeRect(24, 26, 38, 40)));
}

// Checks that PackCropFrame correctly scales location.
TEST(UtilTest, PackKeyFrameInfoScalesLocations) {
  DetectionSet detections;
  AddDetectionFromLocation(MakeRect(10, 10, 10, 10), &detections);
  AddDetectionFromLocation(MakeRect(20, 20, 30, 30), &detections);
  const double scaling = 2.0;
  const int feature_width = kOriginalWidth / scaling;
  const int feature_height = kOriginalHeight / scaling;

  KeyFrameInfo key_frame_info;
  const auto status =
      PackKeyFrameInfo(kTimestamp, detections, kOriginalWidth, kOriginalHeight,
                       feature_width, feature_height, &key_frame_info);

  MP_EXPECT_OK(status);
  EXPECT_EQ(key_frame_info.detections().detections_size(),
            detections.detections_size());
  for (int i = 0; i < detections.detections_size(); ++i) {
    const auto& original_rect = detections.detections(i).location();
    const auto& rect = key_frame_info.detections().detections(i).location();
    EXPECT_TRUE(CheckRectsEqual(rect, original_rect, scaling, scaling));
  }
}

// Checks that PackCropFrame correctly clamps location to be within frame size.
TEST(UtilTest, PackKeyFrameInfoClampsLocations) {
  DetectionSet detections;
  AddDetectionFromLocation(MakeRect(10, 10, 100, 10), &detections);
  AddDetectionFromLocation(MakeRect(0, -10, 110, 100), &detections);
  const int feature_width = kOriginalWidth, feature_height = kOriginalHeight;

  KeyFrameInfo key_frame_info;
  const auto status =
      PackKeyFrameInfo(kTimestamp, detections, kOriginalWidth, kOriginalHeight,
                       feature_width, feature_height, &key_frame_info);

  MP_EXPECT_OK(status);
  EXPECT_EQ(key_frame_info.detections().detections_size(),
            detections.detections_size());
  const auto& out_rect1 = key_frame_info.detections().detections(0).location();
  const auto& out_rect2 = key_frame_info.detections().detections(1).location();
  EXPECT_TRUE(CheckRectsEqual(out_rect1, MakeRect(10, 10, 90, 10)));
  EXPECT_TRUE(CheckRectsEqual(out_rect2, MakeRect(0, 0, 100, 90)));
}

// Checks that PackCropFrame correctly clamps normalized location to be within
// frame size.
TEST(UtilTest, PackKeyFrameInfoClampsNormalizedLocations) {
  DetectionSet detections;
  AddDetectionFromNormalizedLocation(MakeRectF(-0.05, 0.3, 0.4, 0.8),
                                     &detections);
  AddDetectionFromNormalizedLocation(MakeRectF(0.05, -0.1, 1.0, 1.1),
                                     &detections);
  const int feature_width = kOriginalWidth, feature_height = kOriginalHeight;

  KeyFrameInfo key_frame_info;
  const auto status =
      PackKeyFrameInfo(kTimestamp, detections, kOriginalWidth, kOriginalHeight,
                       feature_width, feature_height, &key_frame_info);

  MP_EXPECT_OK(status);
  EXPECT_EQ(key_frame_info.detections().detections_size(),
            detections.detections_size());
  const auto& out_rect1 = key_frame_info.detections().detections(0).location();
  const auto& out_rect2 = key_frame_info.detections().detections(1).location();
  EXPECT_TRUE(CheckRectsEqual(out_rect1, MakeRect(0, 30, 35, 70)));
  EXPECT_TRUE(CheckRectsEqual(out_rect2, MakeRect(5, 0, 95, 100)));
}

// Checks that SortDetections correctly handles empty regions.
TEST(UtilTest, SortDetectionsHandlesEmptyRegions) {
  DetectionSet detections;
  std::vector<SalientRegion> required, non_required;
  MP_EXPECT_OK(SortDetections(detections, &required, &non_required));
  EXPECT_EQ(detections.detections_size(),
            required.size() + non_required.size());
}

// Checks that SortDetections correctly separates required and non-required
// salient regions.
TEST(UtilTest, SortDetectionsSeparatesRequiredAndNonRequiredRegions) {
  DetectionSet detections;
  auto gen_bool = std::bind(std::uniform_int_distribution<>(0, 1),
                            std::default_random_engine());
  for (int i = 0; i < 100; ++i) {
    const bool is_required = gen_bool();
    AddDetectionFromScoreAndIsRequired(1.0, is_required, &detections);
  }

  std::vector<SalientRegion> required, non_required;
  MP_EXPECT_OK(SortDetections(detections, &required, &non_required));
  EXPECT_EQ(detections.detections_size(),
            required.size() + non_required.size());
  for (int i = 0; i < required.size(); ++i) {
    EXPECT_TRUE(required[i].is_required());
  }
  for (int i = 0; i < non_required.size(); ++i) {
    EXPECT_FALSE(non_required[i].is_required());
  }
}

// Checks that SortDetections correctly sorts regions based on scores.
TEST(UtilTest, SortDetectionsSortsRegions) {
  DetectionSet detections;
  auto gen_score = std::bind(std::uniform_real_distribution<>(0.0, 1.0),
                             std::default_random_engine());
  auto gen_bool = std::bind(std::uniform_int_distribution<>(0, 1),
                            std::default_random_engine());
  for (int i = 0; i < 100; ++i) {
    const double score = gen_score();
    const bool is_required = gen_bool();
    AddDetectionFromScoreAndIsRequired(score, is_required, &detections);
  }

  std::vector<SalientRegion> required, non_required;
  MP_EXPECT_OK(SortDetections(detections, &required, &non_required));
  EXPECT_EQ(detections.detections_size(),
            required.size() + non_required.size());
  for (int i = 0; i < required.size() - 1; ++i) {
    EXPECT_GE(required[i].score(), required[i + 1].score());
  }
  for (int i = 0; i < non_required.size() - 1; ++i) {
    EXPECT_GE(non_required[i].score(), non_required[i + 1].score());
  }
}

// Checks that SetKeyFrameCropTarget checks KeyFrameCropOptions is not null.
TEST(UtilTest, SetKeyFrameCropTargetChecksKeyFrameCropOptionsNotNull) {
  const auto status = SetKeyFrameCropTarget(kOriginalWidth, kOriginalHeight,
                                            kTargetAspectRatio, nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              testing::HasSubstr("KeyFrameCropOptions is null."));
}

// Checks that SetKeyFrameCropTarget checks frame size and target aspect ratio
// are valid.
TEST(UtilTest, SetKeyFrameCropTargetChecksFrameSizeAndTargetAspectRatioValid) {
  KeyFrameCropOptions crop_options;
  auto status = SetKeyFrameCropTarget(0, kOriginalHeight, kTargetAspectRatio,
                                      &crop_options);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              testing::HasSubstr("Frame width is non-positive."));

  status =
      SetKeyFrameCropTarget(kOriginalWidth, kOriginalHeight, 0, &crop_options);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              testing::HasSubstr("Target aspect ratio is non-positive."));
}

// Checks that SetKeyFrameCropTarget correctly sets the target crop size.
TEST(UtilTest, SetKeyFrameCropTargetSetsTargetSizeCorrectly) {
  KeyFrameCropOptions crop_options;
  // 1a) square -> square.
  MP_EXPECT_OK(SetKeyFrameCropTarget(101, 101, 1.0, &crop_options));
  EXPECT_EQ(crop_options.target_width(), 101);
  EXPECT_EQ(crop_options.target_height(), 101);
  // 1b) square -> landscape.
  MP_EXPECT_OK(SetKeyFrameCropTarget(101, 101, 1.5, &crop_options));
  EXPECT_EQ(crop_options.target_width(), 101);
  EXPECT_EQ(crop_options.target_height(), 67);
  // 1c) square -> vertical.
  MP_EXPECT_OK(SetKeyFrameCropTarget(101, 101, 0.5, &crop_options));
  EXPECT_EQ(crop_options.target_width(), 51);
  EXPECT_EQ(crop_options.target_height(), 101);
  // 2a) landscape -> square.
  MP_EXPECT_OK(SetKeyFrameCropTarget(128, 72, 1.0, &crop_options));
  EXPECT_EQ(crop_options.target_width(), 72);
  EXPECT_EQ(crop_options.target_height(), 72);
  // 2b) landscape -> more landscape.
  MP_EXPECT_OK(SetKeyFrameCropTarget(128, 72, 2.0, &crop_options));
  EXPECT_EQ(crop_options.target_width(), 128);
  EXPECT_EQ(crop_options.target_height(), 64);
  // 2c) landscape -> vertical.
  MP_EXPECT_OK(SetKeyFrameCropTarget(128, 72, 0.7, &crop_options));
  EXPECT_EQ(crop_options.target_width(), 50);
  EXPECT_EQ(crop_options.target_height(), 72);
  // 3a) vertical -> square.
  MP_EXPECT_OK(SetKeyFrameCropTarget(90, 160, 1.0, &crop_options));
  EXPECT_EQ(crop_options.target_width(), 90);
  EXPECT_EQ(crop_options.target_height(), 90);
  // 3b) vertical -> more vertical.
  MP_EXPECT_OK(SetKeyFrameCropTarget(90, 160, 0.36, &crop_options));
  EXPECT_EQ(crop_options.target_width(), 58);
  EXPECT_EQ(crop_options.target_height(), 160);
  // 3c) vertical -> landscape.
  MP_EXPECT_OK(SetKeyFrameCropTarget(90, 160, 1.2, &crop_options));
  EXPECT_EQ(crop_options.target_width(), 90);
  EXPECT_EQ(crop_options.target_height(), 75);
}

// Checks that AggregateKeyFrameResults checks output pointer is not null.
TEST(UtilTest, AggregateKeyFrameResultsChecksOutputNotNull) {
  const auto status = AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      GetDefaultKeyFrameCropResults(), kOriginalWidth, kOriginalHeight,
      nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              HasSubstr("Output SceneKeyFrameCropSummary is null."));
}

// Checks that AggregateKeyFrameResults handles the case of no key frames.
TEST(UtilTest, AggregateKeyFrameResultsHandlesNoKeyFrames) {
  std::vector<KeyFrameInfo> key_frame_infos(0);
  std::vector<KeyFrameCropResult> key_frame_crop_results(0);
  SceneKeyFrameCropSummary scene_summary;

  MP_EXPECT_OK(AggregateKeyFrameResults(
      key_frame_infos, GetDefaultKeyFrameCropOptions(), key_frame_crop_results,
      kOriginalWidth, kOriginalHeight, &scene_summary));
}

// Checks that AggregateKeyFrameResults checks that number of key frames is
// consistent between KeyFrameInfos and KeyFrameCropResults.
TEST(UtilTest, AggregateKeyFrameResultsChecksNumKeyFramesConsistent) {
  std::vector<KeyFrameInfo> key_frame_infos(kNumKeyFrames);
  std::vector<KeyFrameCropResult> key_frame_crop_results(kNumKeyFrames + 1);
  SceneKeyFrameCropSummary scene_summary;

  const auto status = AggregateKeyFrameResults(
      key_frame_infos, GetDefaultKeyFrameCropOptions(), key_frame_crop_results,
      kOriginalWidth, kOriginalHeight, &scene_summary);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              HasSubstr("Inconsistent number of key frames"));
}

// Checks that AggregateKeyFrameResults checks that frame size is valid.
TEST(UtilTest, AggregateKeyFrameResultsChecksFrameSizeValid) {
  SceneKeyFrameCropSummary scene_summary;
  const auto status = AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      GetDefaultKeyFrameCropResults(), kOriginalWidth, 0, &scene_summary);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Non-positive frame height."));
}

// Checks that AggregateKeyFrameResults checks that target size is valid.
TEST(UtilTest, AggregateKeyFrameResultsChecksTargetSizeValid) {
  KeyFrameCropOptions key_frame_crop_options;
  key_frame_crop_options.set_target_width(0);
  SceneKeyFrameCropSummary scene_summary;

  const auto status = AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), key_frame_crop_options,
      GetDefaultKeyFrameCropResults(), kOriginalWidth, kOriginalHeight,
      &scene_summary);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Non-positive target width."));
}

// Checks that AggregateKeyFrameResults checks that target size does not exceed
// frame size.
TEST(UtilTest, AggregateKeyFrameResultsChecksTargetSizeNotExceedFrameSize) {
  auto key_frame_crop_options = GetDefaultKeyFrameCropOptions();
  key_frame_crop_options.set_target_width(kOriginalWidth + 1);
  SceneKeyFrameCropSummary scene_summary;

  const auto status = AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), key_frame_crop_options,
      GetDefaultKeyFrameCropResults(), kOriginalWidth, kOriginalHeight,
      &scene_summary);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              HasSubstr("Target width exceeds frame width."));
}

// Checks that AggregateKeyFrameResults packs KeyFrameCompactInfos.
TEST(UtilTest, AggregateKeyFrameResultsPacksKeyFrameCompactInfos) {
  const auto key_frame_infos = GetDefaultKeyFrameInfos();
  const auto key_frame_crop_results = GetDefaultKeyFrameCropResults();
  SceneKeyFrameCropSummary scene_summary;

  MP_EXPECT_OK(AggregateKeyFrameResults(
      key_frame_infos, GetDefaultKeyFrameCropOptions(), key_frame_crop_results,
      kOriginalWidth, kOriginalHeight, &scene_summary));

  EXPECT_EQ(scene_summary.num_key_frames(), kNumKeyFrames);
  EXPECT_EQ(scene_summary.key_frame_compact_infos_size(), kNumKeyFrames);
  for (int i = 0; i < kNumKeyFrames; ++i) {
    const auto& compact_info = scene_summary.key_frame_compact_infos(i);
    EXPECT_EQ(compact_info.timestamp_ms(), key_frame_infos[i].timestamp_ms());
    const auto center = RectCenter(key_frame_crop_results[i].region());
    EXPECT_FLOAT_EQ(compact_info.center_x(), center.first);
    EXPECT_FLOAT_EQ(compact_info.center_y(), center.second);
    EXPECT_FLOAT_EQ(compact_info.score(),
                    key_frame_crop_results[i].region_score());
  }
}

// Checks that AggregateKeyFrameResults ensures the centered region of target
// size fits in frame bound.
TEST(UtilTest, AggregateKeyFrameResultsEnsuresCropRegionFitsInFrame) {
  std::vector<KeyFrameInfo> key_frame_infos(1);
  std::vector<KeyFrameCropResult> key_frame_crop_results(1);
  auto* crop_region = key_frame_crop_results[0].mutable_region();
  crop_region->set_x(0);
  crop_region->set_y(0);
  crop_region->set_width(10);
  crop_region->set_height(10);
  SceneKeyFrameCropSummary scene_summary;

  MP_EXPECT_OK(AggregateKeyFrameResults(
      key_frame_infos, GetDefaultKeyFrameCropOptions(), key_frame_crop_results,
      kOriginalWidth, kOriginalHeight, &scene_summary));

  EXPECT_EQ(scene_summary.crop_window_width(), kTargetWidth);
  EXPECT_EQ(scene_summary.crop_window_height(), kTargetHeight);
  const auto& compact_info = scene_summary.key_frame_compact_infos(0);
  const float left = compact_info.center_x() - kTargetWidth / 2.0f;
  const float right = compact_info.center_x() + kTargetWidth / 2.0f;
  const float top = compact_info.center_y() - kTargetWidth / 2.0f;
  const float bottom = compact_info.center_y() + kTargetWidth / 2.0f;
  // Crop window is in the frame.
  EXPECT_GE(left, 0);
  EXPECT_LE(right, kOriginalWidth);
  EXPECT_GE(top, 0);
  EXPECT_LE(bottom, kOriginalHeight);
  // Crop window covers input crop region.
  EXPECT_LE(left, crop_region->x());
  EXPECT_GE(right, crop_region->x() + crop_region->width());
  EXPECT_LE(top, crop_region->y());
  EXPECT_GE(bottom, crop_region->y() + crop_region->height());
}

// Checks that AggregateKeyFrameResults sets centers and scores to -1.0 for key
// frames with empty regions.
TEST(UtilTest,
     AggregateKeyFrameResultsSetsMinusOneForKeyFramesWithEmptyRegions) {
  std::vector<KeyFrameInfo> key_frame_infos(1);
  std::vector<KeyFrameCropResult> key_frame_crop_results(1);
  key_frame_crop_results[0].set_region_is_empty(true);
  SceneKeyFrameCropSummary scene_summary;

  MP_EXPECT_OK(AggregateKeyFrameResults(
      key_frame_infos, GetDefaultKeyFrameCropOptions(), key_frame_crop_results,
      kOriginalWidth, kOriginalHeight, &scene_summary));

  const auto& compact_info = scene_summary.key_frame_compact_infos(0);
  EXPECT_FLOAT_EQ(compact_info.center_x(), -1.0f);
  EXPECT_FLOAT_EQ(compact_info.center_y(), -1.0f);
  EXPECT_FLOAT_EQ(compact_info.score(), -1.0f);
}

// Checks that AggregateKeyFrameResults rejects negative center.
TEST(UtilTest, AggregateKeyFrameResultsRejectsNegativeCenter) {
  auto key_frame_crop_results = GetDefaultKeyFrameCropResults();
  auto* region = key_frame_crop_results[0].mutable_region();
  *region = MakeRectFromCenterAndHalfSize(10, -1.0, 10, 10);
  SceneKeyFrameCropSummary scene_summary;

  const auto status = AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      key_frame_crop_results, kOriginalWidth, kOriginalHeight, &scene_summary);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Negative vertical center."));
}

// Checks that AggregateKeyFrameResults rejects negative score.
TEST(UtilTest, AggregateKeyFrameResultsRejectsNegativeScore) {
  auto key_frame_crop_results = GetDefaultKeyFrameCropResults();
  key_frame_crop_results[0].set_region_score(-1.0);
  SceneKeyFrameCropSummary scene_summary;

  const auto status = AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      key_frame_crop_results, kOriginalWidth, kOriginalHeight, &scene_summary);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Negative score."));
}

// Checks that AggregateKeyFrameResults properly sets center ranges.
TEST(UtilTest, AggregateKeyFrameResultsSetsCenterRanges) {
  auto key_frame_crop_results = GetDefaultKeyFrameCropResults();
  const std::vector<float> centers_x = {30.0, 20.0, 45.0, 3.0, 10.0};
  const std::vector<float> centers_y = {10.0, 0.0, 5.0, 30.0, 20.0};
  const int half_width = 25, half_height = 25;
  for (int i = 0; i < kNumKeyFrames; ++i) {
    auto* region = key_frame_crop_results[i].mutable_region();
    *region = MakeRectFromCenterAndHalfSize(centers_x[i], centers_y[i],
                                            half_width, half_height);
  }
  SceneKeyFrameCropSummary scene_summary;

  MP_EXPECT_OK(AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      key_frame_crop_results, kOriginalWidth, kOriginalHeight, &scene_summary));

  EXPECT_FLOAT_EQ(scene_summary.key_frame_center_min_x(), 25.0f);
  EXPECT_FLOAT_EQ(scene_summary.key_frame_center_max_x(), 45.0f);
  EXPECT_FLOAT_EQ(scene_summary.key_frame_center_min_y(), 25.0f);
  EXPECT_FLOAT_EQ(scene_summary.key_frame_center_max_y(), 30.0f);
}

// Checks that AggregateKeyFrameResults properly sets score range.
TEST(UtilTest, AggregateKeyFrameResultsSetsScoreRange) {
  auto key_frame_crop_results = GetDefaultKeyFrameCropResults();
  const std::vector<float> scores = {0.9, 0.1, 1.2, 2.0, 0.5};
  for (int i = 0; i < kNumKeyFrames; ++i) {
    key_frame_crop_results[i].set_region_score(scores[i]);
  }
  SceneKeyFrameCropSummary scene_summary;

  MP_EXPECT_OK(AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      key_frame_crop_results, kOriginalWidth, kOriginalHeight, &scene_summary));

  EXPECT_FLOAT_EQ(scene_summary.key_frame_min_score(),
                  *std::min_element(scores.begin(), scores.end()));
  EXPECT_FLOAT_EQ(scene_summary.key_frame_max_score(),
                  *std::max_element(scores.begin(), scores.end()));
}

// Checks that AggregateKeyFrameResults sets crop window size to target size
// when the crop regions fit in target size.
TEST(UtilTest, AggregateKeyFrameResultsSetsCropWindowSizeToTargetSize) {
  SceneKeyFrameCropSummary scene_summary;
  MP_EXPECT_OK(AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      GetDefaultKeyFrameCropResults(), kOriginalWidth, kOriginalHeight,
      &scene_summary));
  EXPECT_EQ(scene_summary.crop_window_width(), kTargetWidth);
  EXPECT_EQ(scene_summary.crop_window_height(), kTargetHeight);
}

// Checks that AggregateKeyFrameResults properly sets crop window size when the
// crop regions exceed target size.
TEST(UtilTest, AggregateKeyFrameResultsSetsCropWindowSizeExceedingTargetSize) {
  auto key_frame_crop_results = GetDefaultKeyFrameCropResults();
  key_frame_crop_results[0].mutable_region()->set_width(kTargetWidth + 1);
  SceneKeyFrameCropSummary scene_summary;

  MP_EXPECT_OK(AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      key_frame_crop_results, kOriginalWidth, kOriginalHeight, &scene_summary));
  EXPECT_EQ(scene_summary.crop_window_width(), kTargetWidth + 1);
  EXPECT_EQ(scene_summary.crop_window_height(), kTargetHeight);
}

// Checks that AggregateKeyFrameResults sets has salient region to true when
// there are salient regions.
TEST(UtilTest, AggregateKeyFrameResultsSetsHasSalientRegionTrue) {
  SceneKeyFrameCropSummary scene_summary;
  MP_EXPECT_OK(AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      GetDefaultKeyFrameCropResults(), kOriginalWidth, kOriginalHeight,
      &scene_summary));
  EXPECT_TRUE(scene_summary.has_salient_region());
}

// Checks that AggregateKeyFrameResults sets has salient region to false when
// there are no salient regions.
TEST(UtilTest, AggregateKeyFrameResultsSetsHasSalientRegionFalse) {
  std::vector<KeyFrameCropResult> key_frame_crop_results(kNumKeyFrames);
  for (int i = 0; i < kNumKeyFrames; ++i) {
    key_frame_crop_results[i].set_region_is_empty(true);
  }
  SceneKeyFrameCropSummary scene_summary;

  MP_EXPECT_OK(AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      key_frame_crop_results, kOriginalWidth, kOriginalHeight, &scene_summary));
  EXPECT_FALSE(scene_summary.has_salient_region());
}

// Checks that AggregateKeyFrameResults sets has required salient region to true
// when there are required salient regions.
TEST(UtilTest, AggregateKeyFrameResultsSetsHasRequiredSalientRegionTrue) {
  SceneKeyFrameCropSummary scene_summary;
  MP_EXPECT_OK(AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      GetDefaultKeyFrameCropResults(), kOriginalWidth, kOriginalHeight,
      &scene_summary));
  EXPECT_TRUE(scene_summary.has_required_salient_region());
}

// Checks that AggregateKeyFrameResults sets has required salient region to
// false when there are no required salient regions.
TEST(UtilTest, AggregateKeyFrameResultsSetsHasRequiredSalientRegionFalse) {
  std::vector<KeyFrameCropResult> key_frame_crop_results(kNumKeyFrames);
  for (int i = 0; i < kNumKeyFrames; ++i) {
    key_frame_crop_results[i].set_required_region_is_empty(true);
  }
  SceneKeyFrameCropSummary scene_summary;

  MP_EXPECT_OK(AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      key_frame_crop_results, kOriginalWidth, kOriginalHeight, &scene_summary));
  EXPECT_FALSE(scene_summary.has_required_salient_region());
}

// Checks that AggregateKeyFrameResults sets key frame required crop region
// union.
TEST(UtilTest, AggregateKeyFrameResultsSetsKeyFrameRequiredCropRegionUnion) {
  auto key_frame_crop_results = GetDefaultKeyFrameCropResults();
  for (int i = 0; i < kNumKeyFrames; ++i) {
    *key_frame_crop_results[i].mutable_required_region() =
        MakeRect(i, 0, 50, 50);
  }
  SceneKeyFrameCropSummary scene_summary;

  MP_EXPECT_OK(AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      key_frame_crop_results, kOriginalWidth, kOriginalHeight, &scene_summary));
  const auto& required_crop_region_union =
      scene_summary.key_frame_required_crop_region_union();
  EXPECT_EQ(required_crop_region_union.x(), 0);
  EXPECT_EQ(required_crop_region_union.width(), 50 + kNumKeyFrames - 1);
}

// Checks that AggregateKeyFrameResults properly sets frame success rate.
TEST(UtilTest, AggregateKeyFrameResultsSetsFrameSuccessRate) {
  const int num_success_frames = 3;
  const float success_rate =
      static_cast<float>(num_success_frames) / kNumKeyFrames;
  auto key_frame_crop_results = GetDefaultKeyFrameCropResults();
  for (int i = 0; i < kNumKeyFrames; ++i) {
    const bool successful = i < num_success_frames ? true : false;
    key_frame_crop_results[i].set_are_required_regions_covered_in_target_size(
        successful);
  }
  SceneKeyFrameCropSummary scene_summary;

  MP_EXPECT_OK(AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      key_frame_crop_results, kOriginalWidth, kOriginalHeight, &scene_summary));
  EXPECT_FLOAT_EQ(scene_summary.frame_success_rate(), success_rate);
}

// Checks that AggregateKeyFrameResults properly sets motion.
TEST(UtilTest, AggregateKeyFrameResultsSetsMotion) {
  auto key_frame_crop_results = GetDefaultKeyFrameCropResults();
  const std::vector<float> centers_x = {30.0, 55.0, 45.0, 30.0, 60.0};
  const std::vector<float> centers_y = {30.0, 40.0, 50.0, 45.0, 25.0};
  const float motion_x = (60.0 - 30.0) / kOriginalWidth;
  const float motion_y = (50.0 - 25.0) / kOriginalHeight;
  const int half_width = 25, half_height = 25;
  for (int i = 0; i < kNumKeyFrames; ++i) {
    auto* region = key_frame_crop_results[i].mutable_region();
    *region = MakeRectFromCenterAndHalfSize(centers_x[i], centers_y[i],
                                            half_width, half_height);
  }
  SceneKeyFrameCropSummary scene_summary;

  MP_EXPECT_OK(AggregateKeyFrameResults(
      GetDefaultKeyFrameInfos(), GetDefaultKeyFrameCropOptions(),
      key_frame_crop_results, kOriginalWidth, kOriginalHeight, &scene_summary));
  EXPECT_FLOAT_EQ(scene_summary.horizontal_motion_amount(), motion_x);
  EXPECT_FLOAT_EQ(scene_summary.vertical_motion_amount(), motion_y);
}

// Checks that ComputeSceneStaticBordersSize checks output not null.
TEST(UtilTest, ComputeSceneStaticBordersSizeChecksOutputNotNull) {
  std::vector<StaticFeatures> static_features;
  int bottom_border_size = 0;
  const auto status = ComputeSceneStaticBordersSize(static_features, nullptr,
                                                    &bottom_border_size);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              testing::HasSubstr("Output top border size is null."));
}
// Checks that ComputeSceneStaticBordersSize returns 0 when there are no static
// borders.
TEST(UtilTest, ComputeSceneStaticBordersSizeNoBorder) {
  std::vector<StaticFeatures> static_features(10);
  int top_border_size = 0, bottom_border_size = 0;
  MP_EXPECT_OK(ComputeSceneStaticBordersSize(static_features, &top_border_size,
                                             &bottom_border_size));
  EXPECT_EQ(top_border_size, 0);
  EXPECT_EQ(bottom_border_size, 0);
}

// Checks that ComputeSceneStaticBordersSize correctly computes static border
// size.
TEST(UtilTest, ComputeSceneStaticBordersSizeHasBorders) {
  const int num_frames = 6;
  const std::vector<int> top_borders = {10, 9, 8, 9, 10, 5};
  const std::vector<int> bottom_borders = {7, 7, 7, 7, 6, 7};
  std::vector<StaticFeatures> static_features(num_frames);
  for (int i = 0; i < num_frames; ++i) {
    auto& features = static_features[i];
    auto* top_part = features.add_border();
    top_part->set_relative_position(Border::TOP);
    top_part->mutable_border_position()->set_height(top_borders[i]);
    auto* bottom_part = features.add_border();
    bottom_part->set_relative_position(Border::BOTTOM);
    bottom_part->mutable_border_position()->set_height(bottom_borders[i]);
  }
  int top_border_size = 0, bottom_border_size = 0;
  MP_EXPECT_OK(ComputeSceneStaticBordersSize(static_features, &top_border_size,
                                             &bottom_border_size));
  EXPECT_EQ(top_border_size, 5);
  EXPECT_EQ(bottom_border_size, 6);
}

// Checks that FindSolidBackgroundColor checks output not null.
TEST(UtilTest, FindSolidBackgroundColorChecksOutputNotNull) {
  std::vector<StaticFeatures> static_features;
  std::vector<int64> static_features_timestamps;
  const double min_fraction_solid_background_color = 0.8;
  bool has_solid_background_color;
  PiecewiseLinearFunction l_function, a_function, b_function;

  auto status =
      FindSolidBackgroundColor(static_features, static_features_timestamps,
                               min_fraction_solid_background_color, nullptr,
                               &l_function, &a_function, &b_function);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), testing::HasSubstr("Output boolean is null."));

  status = FindSolidBackgroundColor(static_features, static_features_timestamps,
                                    min_fraction_solid_background_color,
                                    &has_solid_background_color, nullptr,
                                    &a_function, &b_function);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              testing::HasSubstr("Output color l function is null."));
}

// Checks that FindSolidBackgroundColor returns true when there is solid
// background color.
TEST(UtilTest, FindSolidBackgroundColorReturnsTrue) {
  std::vector<StaticFeatures> static_features(1);
  auto* color = static_features[0].mutable_solid_background();
  color->set_r(255);
  color->set_g(255);
  color->set_b(255);
  std::vector<int64> static_features_timestamps(1);
  static_features_timestamps[0] = 0;
  const double min_fraction_solid_background_color = 0.8;
  bool has_solid_background_color;
  PiecewiseLinearFunction l_function, a_function, b_function;

  MP_EXPECT_OK(FindSolidBackgroundColor(
      static_features, static_features_timestamps,
      min_fraction_solid_background_color, &has_solid_background_color,
      &l_function, &a_function, &b_function));
  EXPECT_TRUE(has_solid_background_color);
}

// Checks that FindSolidBackgroundColor returns false when there is no solid
// background color.
TEST(UtilTest, FindSolidBackgroundColorReturnsFalse) {
  std::vector<StaticFeatures> static_features(1);
  std::vector<int64> static_features_timestamps(1);
  static_features_timestamps[0] = 0;
  const double min_fraction_solid_background_color = 0.8;
  bool has_solid_background_color;
  PiecewiseLinearFunction l_function, a_function, b_function;

  MP_EXPECT_OK(FindSolidBackgroundColor(
      static_features, static_features_timestamps,
      min_fraction_solid_background_color, &has_solid_background_color,
      &l_function, &a_function, &b_function));
  EXPECT_FALSE(has_solid_background_color);
}

// Checks that FindSolidBackgroundColor sets the interpolation functions.
TEST(UtilTest, FindSolidBackgroundColorSetsInterpolationFunctions) {
  const uint8 rgb1[] = {255, 255, 0};  // cyan in bgr
  const double lab1[] = {91.1133, -48.0938, -14.125};
  const int64 time1 = 0;
  const uint8 rgb2[] = {255, 0, 255};  // magenta in bgr
  const double lab2[] = {60.321, 98.2344, -60.8281};
  const int64 time2 = 2000;
  std::vector<StaticFeatures> static_features(2);
  auto* color1 = static_features[0].mutable_solid_background();
  color1->set_r(rgb1[0]);
  color1->set_g(rgb1[1]);
  color1->set_b(rgb1[2]);
  auto* color2 = static_features[1].mutable_solid_background();
  color2->set_r(rgb2[0]);
  color2->set_g(rgb2[1]);
  color2->set_b(rgb2[2]);
  std::vector<int64> static_features_timestamps(2);
  static_features_timestamps[0] = time1;
  static_features_timestamps[1] = time2;
  const double min_fraction_solid_background_color = 0.8;
  bool has_solid_background_color;
  PiecewiseLinearFunction l_function, a_function, b_function;

  MP_EXPECT_OK(FindSolidBackgroundColor(
      static_features, static_features_timestamps,
      min_fraction_solid_background_color, &has_solid_background_color,
      &l_function, &a_function, &b_function));

  EXPECT_TRUE(has_solid_background_color);
  EXPECT_LE(std::fabs(l_function.Evaluate(time1) - lab1[0]), 1e-2f);
  EXPECT_LE(std::fabs(a_function.Evaluate(time1) - lab1[1]), 1e-2f);
  EXPECT_LE(std::fabs(b_function.Evaluate(time1) - lab1[2]), 1e-2f);
  EXPECT_LE(std::fabs(l_function.Evaluate(time2) - lab2[0]), 1e-2f);
  EXPECT_LE(std::fabs(a_function.Evaluate(time2) - lab2[1]), 1e-2f);
  EXPECT_LE(std::fabs(b_function.Evaluate(time2) - lab2[2]), 1e-2f);

  EXPECT_LE(std::fabs(l_function.Evaluate((time1 + time2) / 2.0) -
                      (lab1[0] + lab2[0]) / 2.0),
            1e-2f);
  EXPECT_LE(std::fabs(a_function.Evaluate((time1 + time2) / 2.0) -
                      (lab1[1] + lab2[1]) / 2.0),
            1e-2f);
  EXPECT_LE(std::fabs(b_function.Evaluate((time1 + time2) / 2.0) -
                      (lab1[2] + lab2[2]) / 2.0),
            1e-2f);
}

TEST(UtilTest, TestAffineRetargeterPass) {
  std::vector<cv::Mat> transforms;
  std::vector<cv::Mat> frames;
  std::vector<cv::Mat> results;
  for (int i = 0; i < 5; i++) {
    cv::Mat transform = cv::Mat(2, 3, CV_32FC1);
    transform.at<float>(0, 0) = 1;
    transform.at<float>(0, 1) = 0;
    transform.at<float>(1, 0) = 0;
    transform.at<float>(1, 1) = 1;
    transform.at<float>(0, 2) = -i * 50;
    transform.at<float>(1, 2) = 0;
    transforms.push_back(transform);
    cv::Mat image = cv::Mat::zeros(1080, 1920, CV_8UC3);
    cv::Vec3b val;
    val[0] = 255;
    val[1] = 255;
    val[2] = 255;
    image(cv::Rect(0, 0, 395, 1080)).setTo(val);
    frames.push_back(image);

    cv::Mat image_out = cv::Mat::zeros(500, 1920, CV_8UC3);
    results.push_back(image_out);
  }

  MP_ASSERT_OK(
      AffineRetarget(cv::Size(500, 1080), frames, transforms, &results));
  ASSERT_EQ(results.size(), 5);
  for (int i = 0; i < 5; i++) {
    EXPECT_GT(results[i].at<cv::Vec3b>(540, 390 - i * 50)[0], 250);
    EXPECT_GT(results[i].at<cv::Vec3b>(540, 390 - i * 50)[1], 250);
    EXPECT_GT(results[i].at<cv::Vec3b>(540, 390 - i * 50)[2], 250);
    EXPECT_LT(results[i].at<cv::Vec3b>(540, 400 - i * 50)[0], 5);
    EXPECT_LT(results[i].at<cv::Vec3b>(540, 400 - i * 50)[1], 5);
    EXPECT_LT(results[i].at<cv::Vec3b>(540, 400 - i * 50)[2], 5);
  }
}

TEST(UtilTest, TestAffineRetargeterFail) {
  std::vector<cv::Mat> transforms;
  std::vector<cv::Mat> frames;
  std::vector<cv::Mat> results;

  cv::Mat dummy;
  transforms.push_back(dummy);
  frames.push_back(dummy);

  EXPECT_THAT(
      AffineRetarget(cv::Size(500, 1080), frames, transforms, &results)
          .ToString(),
      testing::HasSubstr("Output vector cropped_frames must be populated"));
}

}  // namespace
}  // namespace autoflip
}  // namespace mediapipe
