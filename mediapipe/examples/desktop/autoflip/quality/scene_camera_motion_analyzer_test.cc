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

#include "mediapipe/examples/desktop/autoflip/quality/scene_camera_motion_analyzer.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/focus_point.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/piecewise_linear_function.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace autoflip {

using ::testing::HasSubstr;

const int kNumKeyFrames = 5;
const int kNumSceneFrames = 30;

const int64 kKeyFrameTimestampDiff = 1e6 / kNumKeyFrames;
const int64 kSceneFrameTimestampDiff = 1e6 / kNumSceneFrames;
// Default time span of a scene in seconds.
const double kSceneTimeSpanSec = 1.0;

const int kSceneFrameWidth = 100;
const int kSceneFrameHeight = 100;

const int kTargetWidth = 50;
const int kTargetHeight = 50;

constexpr char kCameraTrackingSceneFrameResultsFile[] =
    "mediapipe/examples/desktop/autoflip/quality/testdata/"
    "camera_motion_tracking_scene_frame_results.csv";

// Makes a rectangle given the corner (x, y) and the size (width, height).
Rect MakeRect(const int x, const int y, const int width, const int height) {
  Rect rect;
  rect.set_x(x);
  rect.set_y(y);
  rect.set_width(width);
  rect.set_height(height);
  return rect;
}

// Returns default values for scene frame timestamps. Populates timestamps using
// the default spacing kSceneFrameTimestampDiff starting from 0.
std::vector<int64> GetDefaultSceneFrameTimestamps() {
  std::vector<int64> scene_frame_timestamps(kNumSceneFrames);
  for (int i = 0; i < kNumSceneFrames; ++i) {
    scene_frame_timestamps[i] = kSceneFrameTimestampDiff * i;
  }
  return scene_frame_timestamps;
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
    key_frame_crop_results[i].set_timestamp_ms(kKeyFrameTimestampDiff * i);
  }
  return key_frame_crop_results;
}

// Returns default settings for SceneKeyFrameCropSummary. Sets scene frame size
// to be the default size. Sets each key frame compact info in accordance to the
// default timestamps (using the default spacing kKeyFrameTimestampDiff starting
// from 0), default crop regions (centered at (25, 25)), and default scores
// (1.0). Sets center range to be [25, 25] and [25, 25]. Sets score range to be
// [1.0, 1.0]. Sets crop window size to be (25, 25). Sets has focus region to
// be true. Sets frame success rate to be 1.0. Sets horizontal and vertical
// motion amount to be 0.0.
SceneKeyFrameCropSummary GetDefaultSceneKeyFrameCropSummary() {
  SceneKeyFrameCropSummary scene_summary;
  scene_summary.set_scene_frame_width(kSceneFrameWidth);
  scene_summary.set_scene_frame_height(kSceneFrameHeight);
  scene_summary.set_num_key_frames(kNumKeyFrames);
  for (int i = 0; i < kNumKeyFrames; ++i) {
    auto* compact_info = scene_summary.add_key_frame_compact_infos();
    compact_info->set_timestamp_ms(kKeyFrameTimestampDiff * i);
    compact_info->set_center_x(25);
    compact_info->set_center_y(25);
    compact_info->set_score(1.0);
  }
  scene_summary.set_key_frame_center_min_x(25);
  scene_summary.set_key_frame_center_max_x(25);
  scene_summary.set_key_frame_center_min_y(25);
  scene_summary.set_key_frame_center_max_y(25);
  scene_summary.set_key_frame_min_score(1.0);
  scene_summary.set_key_frame_max_score(1.0);
  scene_summary.set_crop_window_width(25);
  scene_summary.set_crop_window_height(25);
  scene_summary.set_has_salient_region(true);
  scene_summary.set_frame_success_rate(1.0);
  scene_summary.set_horizontal_motion_amount(0.0);
  scene_summary.set_vertical_motion_amount(0.0);
  return scene_summary;
}

// Returns a SceneKeyFrameCropSummary with small motion. Sets crop window size
// to default target size. Sets horizontal motion to half the threshold in the
// options and vertical motion to 0. Sets center x range to [45, 55].
SceneKeyFrameCropSummary GetSceneKeyFrameCropSummaryWithSmallMotion(
    const SceneCameraMotionAnalyzerOptions& options) {
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  scene_summary.set_crop_window_width(kTargetWidth);
  scene_summary.set_crop_window_height(kTargetHeight);
  scene_summary.set_horizontal_motion_amount(
      options.motion_stabilization_threshold_percent() / 2.0);
  scene_summary.set_vertical_motion_amount(0.0);
  scene_summary.set_key_frame_center_min_x(45);
  scene_summary.set_key_frame_center_max_x(55);
  return scene_summary;
}

// Testable class that allows public access to protected methods in the class.
class TestableSceneCameraMotionAnalyzer : public SceneCameraMotionAnalyzer {
 public:
  explicit TestableSceneCameraMotionAnalyzer(
      const SceneCameraMotionAnalyzerOptions&
          scene_camera_motion_analyzer_options)
      : SceneCameraMotionAnalyzer(scene_camera_motion_analyzer_options) {}
  ~TestableSceneCameraMotionAnalyzer() {}
  using SceneCameraMotionAnalyzer::DecideCameraMotionType;
  using SceneCameraMotionAnalyzer::PopulateFocusPointFrames;
};

// Checks that DecideCameraMotionType checks that output pointers are not null.
TEST(SceneCameraMotionAnalyzerTest, DecideCameraMotionTypeChecksOutputNotNull) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  KeyFrameCropOptions crop_options = GetDefaultKeyFrameCropOptions();
  SceneKeyFrameCropSummary scene_summary;
  SceneCameraMotion camera_motion;
  auto status = analyzer.DecideCameraMotionType(crop_options, kSceneTimeSpanSec,
                                                0, nullptr, &camera_motion);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Scene summary is null."));
  status = analyzer.DecideCameraMotionType(crop_options, kSceneTimeSpanSec, 0,
                                           &scene_summary, nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Scene camera motion is null."));
}

// Checks that DecideCameraMotionType properly handles the case where no key
// frame has any focus region, and sets the camera motion type to steady and
// the look-at position to the scene frame center.
TEST(SceneCameraMotionAnalyzerTest,
     DecideCameraMotionTypeWithoutAnyFocusRegion) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  KeyFrameCropOptions crop_options = GetDefaultKeyFrameCropOptions();
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  scene_summary.set_has_salient_region(false);
  SceneCameraMotion camera_motion;

  MP_EXPECT_OK(analyzer.DecideCameraMotionType(
      crop_options, kSceneTimeSpanSec, 0, &scene_summary, &camera_motion));
  EXPECT_TRUE(camera_motion.has_steady_motion());
  const auto& steady_motion = camera_motion.steady_motion();
  EXPECT_FLOAT_EQ(steady_motion.steady_look_at_center_x(),
                  kSceneFrameWidth / 2.0f);
  EXPECT_FLOAT_EQ(steady_motion.steady_look_at_center_y(),
                  kSceneFrameHeight / 2.0f);
}

// Checks that DecideCameraMotionType properly handles the camera sweeps from
// left to right.
TEST(SceneCameraMotionAnalyzerTest, DecideCameraMotionTypeSweepingLeftToRight) {
  SceneCameraMotionAnalyzerOptions options;
  options.set_sweep_entire_frame(true);
  TestableSceneCameraMotionAnalyzer analyzer(options);
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  scene_summary.set_frame_success_rate(
      options.minimum_success_rate_for_sweeping() / 2.0f);
  scene_summary.set_crop_window_width(kTargetWidth * 1.5f);  // horizontal sweep
  scene_summary.set_crop_window_height(kTargetHeight);
  const double time_span = options.minimum_scene_span_sec_for_sweeping() * 2.0;
  SceneCameraMotion camera_motion;

  MP_EXPECT_OK(analyzer.DecideCameraMotionType(GetDefaultKeyFrameCropOptions(),
                                               time_span, 0, &scene_summary,
                                               &camera_motion));

  EXPECT_TRUE(camera_motion.has_sweeping_motion());
  const auto& sweeping_motion = camera_motion.sweeping_motion();
  EXPECT_FLOAT_EQ(sweeping_motion.sweep_start_center_x(), 0.0f);
  EXPECT_FLOAT_EQ(sweeping_motion.sweep_end_center_x(),
                  static_cast<float>(kSceneFrameWidth));
  EXPECT_FLOAT_EQ(sweeping_motion.sweep_start_center_y(),
                  kSceneFrameHeight / 2.0f);
  EXPECT_FLOAT_EQ(sweeping_motion.sweep_end_center_y(),
                  kSceneFrameHeight / 2.0f);
}

// Checks that DecideCameraMotionType properly handles the camera sweeps from
// top to bottom.
TEST(SceneCameraMotionAnalyzerTest, DecideCameraMotionTypeSweepingTopToBottom) {
  SceneCameraMotionAnalyzerOptions options;
  options.set_sweep_entire_frame(true);
  TestableSceneCameraMotionAnalyzer analyzer(options);
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  scene_summary.set_frame_success_rate(
      options.minimum_success_rate_for_sweeping() / 2.0f);
  scene_summary.set_crop_window_width(kTargetWidth);
  scene_summary.set_crop_window_height(kTargetHeight * 1.5f);  // vertical sweep
  const double time_span = options.minimum_scene_span_sec_for_sweeping() * 2.0;
  SceneCameraMotion camera_motion;

  MP_EXPECT_OK(analyzer.DecideCameraMotionType(GetDefaultKeyFrameCropOptions(),
                                               time_span, 0, &scene_summary,
                                               &camera_motion));

  EXPECT_TRUE(camera_motion.has_sweeping_motion());
  const auto& sweeping_motion = camera_motion.sweeping_motion();
  EXPECT_FLOAT_EQ(sweeping_motion.sweep_start_center_y(), 0.0f);
  EXPECT_FLOAT_EQ(sweeping_motion.sweep_end_center_y(),
                  static_cast<float>(kSceneFrameWidth));
  EXPECT_FLOAT_EQ(sweeping_motion.sweep_start_center_x(),
                  kSceneFrameWidth / 2.0f);
  EXPECT_FLOAT_EQ(sweeping_motion.sweep_end_center_x(),
                  kSceneFrameWidth / 2.0f);
}

// Checks that DecideCameraMotionType properly handles the camera sweeps from
// one corner of the center range to another.
TEST(SceneCameraMotionAnalyzerTest, DecideCameraMotionTypeSweepingCenterRange) {
  SceneCameraMotionAnalyzerOptions options;
  options.set_sweep_entire_frame(false);
  TestableSceneCameraMotionAnalyzer analyzer(options);
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  scene_summary.set_frame_success_rate(
      options.minimum_success_rate_for_sweeping() / 2.0f);
  scene_summary.set_crop_window_width(kTargetWidth * 1.5f);
  scene_summary.set_crop_window_height(kTargetHeight * 1.5f);
  const double time_span = options.minimum_scene_span_sec_for_sweeping() * 2.0;
  SceneCameraMotion camera_motion;

  MP_EXPECT_OK(analyzer.DecideCameraMotionType(GetDefaultKeyFrameCropOptions(),
                                               time_span, 0, &scene_summary,
                                               &camera_motion));

  EXPECT_TRUE(camera_motion.has_sweeping_motion());
  const auto& sweeping_motion = camera_motion.sweeping_motion();
  EXPECT_FLOAT_EQ(sweeping_motion.sweep_start_center_x(),
                  scene_summary.key_frame_center_min_x());
  EXPECT_FLOAT_EQ(sweeping_motion.sweep_start_center_y(),
                  scene_summary.key_frame_center_min_y());
  EXPECT_FLOAT_EQ(sweeping_motion.sweep_end_center_x(),
                  scene_summary.key_frame_center_max_x());
  EXPECT_FLOAT_EQ(sweeping_motion.sweep_end_center_y(),
                  scene_summary.key_frame_center_max_y());
}

// Checks that DecideCameraMotionType properly handles the case where motion is
// small and there are no required focus regions.
TEST(SceneCameraMotionAnalyzerTest,
     DecideCameraMotionTypeSmallMotionNoRequiredFocusRegion) {
  SceneCameraMotionAnalyzerOptions options;
  options.set_motion_stabilization_threshold_percent(0.1);
  options.set_snap_center_max_distance_percent(0.0);
  TestableSceneCameraMotionAnalyzer analyzer(options);
  auto scene_summary = GetSceneKeyFrameCropSummaryWithSmallMotion(options);
  scene_summary.set_has_required_salient_region(false);
  const int crop_region_center_x = 50;
  SceneCameraMotion camera_motion;

  MP_EXPECT_OK(analyzer.DecideCameraMotionType(GetDefaultKeyFrameCropOptions(),
                                               kSceneTimeSpanSec, 0,
                                               &scene_summary, &camera_motion));
  EXPECT_TRUE(camera_motion.has_steady_motion());
  EXPECT_EQ(camera_motion.steady_motion().steady_look_at_center_x(),
            crop_region_center_x);
  EXPECT_EQ(scene_summary.crop_window_width(), kTargetWidth);
  EXPECT_EQ(scene_summary.crop_window_height(), kTargetHeight);
}

// Checks that DecideCameraMotionType properly handles the case where motion is
// small and there are required focus regions that fit in target size.
TEST(SceneCameraMotionAnalyzerTest,
     DecideCameraMotionTypeSmallMotionRequiredFocusRegionInTargetSize) {
  SceneCameraMotionAnalyzerOptions options;
  options.set_motion_stabilization_threshold_percent(0.1);
  options.set_snap_center_max_distance_percent(0.0);
  TestableSceneCameraMotionAnalyzer analyzer(options);
  auto scene_summary = GetSceneKeyFrameCropSummaryWithSmallMotion(options);
  scene_summary.set_has_required_salient_region(true);
  *scene_summary.mutable_key_frame_required_crop_region_union() =
      MakeRect(40, 0, 40, 10);
  const int required_region_center_x = 60;
  SceneCameraMotion camera_motion;

  MP_EXPECT_OK(analyzer.DecideCameraMotionType(GetDefaultKeyFrameCropOptions(),
                                               kSceneTimeSpanSec, 0,
                                               &scene_summary, &camera_motion));
  EXPECT_TRUE(camera_motion.has_steady_motion());
  EXPECT_EQ(camera_motion.steady_motion().steady_look_at_center_x(),
            required_region_center_x);
  EXPECT_EQ(scene_summary.crop_window_width(), 50);
  EXPECT_EQ(scene_summary.crop_window_height(), kTargetHeight);
}

// Checks that DecideCameraMotionType properly handles the case where motion is
// small and there are required focus regions that exceed target size.
TEST(SceneCameraMotionAnalyzerTest,
     DecideCameraMotionTypeSmallMotionRequiredFocusRegionExceedingTargetSize) {
  SceneCameraMotionAnalyzerOptions options;
  options.set_motion_stabilization_threshold_percent(0.1);
  options.set_snap_center_max_distance_percent(0.0);
  TestableSceneCameraMotionAnalyzer analyzer(options);
  auto scene_summary = GetSceneKeyFrameCropSummaryWithSmallMotion(options);
  scene_summary.set_has_required_salient_region(true);
  *scene_summary.mutable_key_frame_required_crop_region_union() =
      MakeRect(20, 0, 70, 10);
  const int required_region_center_x = 55;
  SceneCameraMotion camera_motion;

  MP_EXPECT_OK(analyzer.DecideCameraMotionType(GetDefaultKeyFrameCropOptions(),
                                               kSceneTimeSpanSec, 0,
                                               &scene_summary, &camera_motion));
  EXPECT_TRUE(camera_motion.has_steady_motion());
  EXPECT_EQ(camera_motion.steady_motion().steady_look_at_center_x(),
            required_region_center_x);
  EXPECT_EQ(scene_summary.crop_window_width(), 70);
  EXPECT_EQ(scene_summary.crop_window_height(), kTargetHeight);
}

// Checks that DecideCameraMotionType properly handles the case where motion is
// small and the middle of the key frame crop center range is close to the scene
// frame center.
TEST(SceneCameraMotionAnalyzerTest,
     DecideCameraMotionTypeSmallMotionCloseToCenter) {
  SceneCameraMotionAnalyzerOptions options;
  options.set_motion_stabilization_threshold_percent(0.1);
  options.set_snap_center_max_distance_percent(0.1);
  TestableSceneCameraMotionAnalyzer analyzer(options);
  KeyFrameCropOptions crop_options = GetDefaultKeyFrameCropOptions();
  const float frame_center_x = kSceneFrameWidth / 2.0f;
  auto scene_summary = GetSceneKeyFrameCropSummaryWithSmallMotion(options);
  scene_summary.set_key_frame_center_min_x(frame_center_x - 2);
  scene_summary.set_key_frame_center_max_x(frame_center_x);
  SceneCameraMotion camera_motion;

  MP_EXPECT_OK(analyzer.DecideCameraMotionType(
      crop_options, kSceneTimeSpanSec, 0, &scene_summary, &camera_motion));
  EXPECT_TRUE(camera_motion.has_steady_motion());
  EXPECT_FLOAT_EQ(camera_motion.steady_motion().steady_look_at_center_x(),
                  frame_center_x);
}

// Checks that DecideCameraMotionType properly handles the case where motion is
// not small, and sets the camera motion type to tracking.
TEST(SceneCameraMotionAnalyzerTest, DecideCameraMotionTypeTracking) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  scene_summary.set_horizontal_motion_amount(
      options.motion_stabilization_threshold_percent() * 2.0);
  SceneCameraMotion camera_motion;

  MP_EXPECT_OK(analyzer.DecideCameraMotionType(GetDefaultKeyFrameCropOptions(),
                                               kSceneTimeSpanSec, 0,
                                               &scene_summary, &camera_motion));
  EXPECT_TRUE(camera_motion.has_tracking_motion());
}

// Checks that PopulateFocusPointFrames checks output pointer is not null.
TEST(SceneCameraMotionAnalyzerTest,
     PopulateFocusPointFramesChecksOutputNotNull) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  const auto status = analyzer.PopulateFocusPointFrames(
      GetDefaultSceneKeyFrameCropSummary(), camera_motion,
      GetDefaultSceneFrameTimestamps(), nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              HasSubstr("Output vector of FocusPointFrame is null."));
}

// Checks that PopulateFocusPointFrames checks scene frames size.
TEST(SceneCameraMotionAnalyzerTest,
     PopulateFocusPointFramesChecksSceneFramesSize) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  std::vector<int64> scene_frame_timestamps(0);
  std::vector<FocusPointFrame> focus_point_frames;

  const auto status = analyzer.PopulateFocusPointFrames(
      GetDefaultSceneKeyFrameCropSummary(), camera_motion,
      scene_frame_timestamps, &focus_point_frames);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("No scene frames."));
}

// Checks that PopulateFocusPointFrames handles the case of no key frames.
TEST(SceneCameraMotionAnalyzerTest,
     PopulateFocusPointFramesHandlesNoKeyFrames) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  camera_motion.mutable_steady_motion();
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  scene_summary.set_num_key_frames(0);
  scene_summary.clear_key_frame_compact_infos();
  scene_summary.set_has_salient_region(false);
  std::vector<FocusPointFrame> focus_point_frames;
  MP_EXPECT_OK(analyzer.PopulateFocusPointFrames(
      scene_summary, camera_motion, GetDefaultSceneFrameTimestamps(),
      &focus_point_frames));
}

// Checks that PopulateFocusPointFrames checks KeyFrameCompactInfos has the
// right size.
TEST(SceneCameraMotionAnalyzerTest,
     PopulateFocusPointFramesChecksKeyFrameCompactInfosSize) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  scene_summary.set_num_key_frames(2 * kNumKeyFrames);
  std::vector<FocusPointFrame> focus_point_frames;

  const auto status = analyzer.PopulateFocusPointFrames(
      scene_summary, camera_motion, GetDefaultSceneFrameTimestamps(),
      &focus_point_frames);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              HasSubstr("Key frame compact infos has wrong size"));
}

// Checks that PopulateFocusPointFrames checks SceneKeyFrameCropSummary has
// valid scene frame size.
TEST(SceneCameraMotionAnalyzerTest,
     PopulateFocusPointFramesChecksSceneFrameSize) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  scene_summary.set_scene_frame_height(0);
  std::vector<FocusPointFrame> focus_point_frames;

  const auto status = analyzer.PopulateFocusPointFrames(
      scene_summary, camera_motion, GetDefaultSceneFrameTimestamps(),
      &focus_point_frames);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Non-positive frame height."));
}

// Checks that PopulateFocusPointFrames checks camera motion type is valid.
TEST(SceneCameraMotionAnalyzerTest,
     PopulateFocusPointFramesChecksCameraMotionType) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  camera_motion.clear_motion_type();
  std::vector<FocusPointFrame> focus_point_frames;

  const auto status = analyzer.PopulateFocusPointFrames(
      GetDefaultSceneKeyFrameCropSummary(), camera_motion,
      GetDefaultSceneFrameTimestamps(), &focus_point_frames);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Unknown motion type."));
}

// Checks that PopulateFocusPointFrames properly sets FocusPointFrames when
// camera motion type is steady.
TEST(SceneCameraMotionAnalyzerTest, PopulateFocusPointFramesSteady) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  auto* steady_motion = camera_motion.mutable_steady_motion();
  steady_motion->set_steady_look_at_center_x(40.5);
  steady_motion->set_steady_look_at_center_y(25);
  std::vector<FocusPointFrame> focus_point_frames;

  MP_EXPECT_OK(analyzer.PopulateFocusPointFrames(
      GetDefaultSceneKeyFrameCropSummary(), camera_motion,
      GetDefaultSceneFrameTimestamps(), &focus_point_frames));

  EXPECT_EQ(kNumSceneFrames, focus_point_frames.size());
  for (int i = 0; i < kNumSceneFrames; ++i) {
    // FocusPointFrameType is CENTER.
    EXPECT_EQ(focus_point_frames[i].point_size(), 1);
    const auto& point = focus_point_frames[i].point(0);
    EXPECT_FLOAT_EQ(
        point.norm_point_x(),
        steady_motion->steady_look_at_center_x() / kSceneFrameWidth);
    EXPECT_FLOAT_EQ(
        point.norm_point_y(),
        steady_motion->steady_look_at_center_y() / kSceneFrameHeight);
    EXPECT_FLOAT_EQ(point.weight(), options.maximum_salient_point_weight());
  }
}

// Checks that PopulateFocusPointFrames properly sets FocusPointFrames when
// FocusPointFrameType is TOPMOST_AND_BOTTOMMOST.
TEST(SceneCameraMotionAnalyzerTest, PopulateFocusPointFramesTopAndBottom) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  auto* steady_motion = camera_motion.mutable_steady_motion();
  steady_motion->set_steady_look_at_center_x(40.5);
  steady_motion->set_steady_look_at_center_y(25);
  // Forces FocusPointFrameType to be TOPMOST_AND_BOTTOMOST.
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  scene_summary.set_crop_window_height(kSceneFrameHeight);
  std::vector<FocusPointFrame> focus_point_frames;

  MP_EXPECT_OK(analyzer.PopulateFocusPointFrames(
      scene_summary, camera_motion, GetDefaultSceneFrameTimestamps(),
      &focus_point_frames));

  EXPECT_EQ(kNumSceneFrames, focus_point_frames.size());
  for (int i = 0; i < kNumSceneFrames; ++i) {
    EXPECT_EQ(focus_point_frames[i].point_size(), 2);
    const auto& point1 = focus_point_frames[i].point(0);
    const auto& point2 = focus_point_frames[i].point(1);
    EXPECT_FLOAT_EQ(
        point1.norm_point_x(),
        steady_motion->steady_look_at_center_x() / kSceneFrameWidth);
    EXPECT_FLOAT_EQ(point1.norm_point_y(), 0.0f);
    EXPECT_FLOAT_EQ(
        point2.norm_point_x(),
        steady_motion->steady_look_at_center_x() / kSceneFrameWidth);
    EXPECT_FLOAT_EQ(point2.norm_point_y(), 1.0f);
    EXPECT_FLOAT_EQ(point1.weight(), options.maximum_salient_point_weight());
    EXPECT_FLOAT_EQ(point2.weight(), options.maximum_salient_point_weight());
  }
}

// Checks that PopulateFocusPointFrames properly sets FocusPointFrames when
// FocusPointFrameType is LEFTMOST_AND_RIGHTMOST.
TEST(SceneCameraMotionAnalyzerTest, PopulateFocusPointFramesLeftAndRight) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  auto* steady_motion = camera_motion.mutable_steady_motion();
  steady_motion->set_steady_look_at_center_x(40.5);
  steady_motion->set_steady_look_at_center_y(25);
  // Forces FocusPointFrameType to be LEFTMOST_AND_RIGHTMOST.
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  scene_summary.set_crop_window_width(kSceneFrameWidth);
  std::vector<FocusPointFrame> focus_point_frames;

  MP_EXPECT_OK(analyzer.PopulateFocusPointFrames(
      scene_summary, camera_motion, GetDefaultSceneFrameTimestamps(),
      &focus_point_frames));

  EXPECT_EQ(kNumSceneFrames, focus_point_frames.size());
  for (int i = 0; i < kNumSceneFrames; ++i) {
    EXPECT_EQ(focus_point_frames[i].point_size(), 2);
    const auto& point1 = focus_point_frames[i].point(0);
    const auto& point2 = focus_point_frames[i].point(1);
    EXPECT_FLOAT_EQ(point1.norm_point_x(), 0.0f);
    EXPECT_FLOAT_EQ(
        point1.norm_point_y(),
        steady_motion->steady_look_at_center_y() / kSceneFrameHeight);
    EXPECT_FLOAT_EQ(point2.norm_point_x(), 1.0f);
    EXPECT_FLOAT_EQ(
        point1.norm_point_y(),
        steady_motion->steady_look_at_center_y() / kSceneFrameHeight);
    EXPECT_FLOAT_EQ(point1.weight(), options.maximum_salient_point_weight());
    EXPECT_FLOAT_EQ(point2.weight(), options.maximum_salient_point_weight());
  }
}

// Checks that PopulateFocusPointFrames properly sets FocusPointFrames when
// camera motion type is sweeping.
TEST(SceneCameraMotionAnalyzerTest, PopulateFocusPointFramesSweeping) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  auto* sweeping_motion = camera_motion.mutable_sweeping_motion();
  sweeping_motion->set_sweep_start_center_x(5);
  sweeping_motion->set_sweep_start_center_y(50);
  sweeping_motion->set_sweep_end_center_x(95);
  sweeping_motion->set_sweep_end_center_y(50);
  const int num_frames = 10;
  const std::vector<float> positions_x = {5,  15, 25, 35, 45,
                                          55, 65, 75, 85, 95};
  const std::vector<float> positions_y = {50, 50, 50, 50, 50,
                                          50, 50, 50, 50, 50};
  std::vector<int64> scene_frame_timestamps(num_frames);
  std::iota(scene_frame_timestamps.begin(), scene_frame_timestamps.end(), 0);
  std::vector<FocusPointFrame> focus_point_frames;

  MP_EXPECT_OK(analyzer.PopulateFocusPointFrames(
      GetDefaultSceneKeyFrameCropSummary(), camera_motion,
      scene_frame_timestamps, &focus_point_frames));

  EXPECT_EQ(num_frames, focus_point_frames.size());
  for (int i = 0; i < num_frames; ++i) {
    EXPECT_EQ(focus_point_frames[i].point_size(), 1);
    const auto& point = focus_point_frames[i].point(0);
    EXPECT_FLOAT_EQ(positions_x[i] / kSceneFrameWidth, point.norm_point_x());
    EXPECT_FLOAT_EQ(positions_y[i] / kSceneFrameHeight, point.norm_point_y());
  }
}

// Checks that PopulateFocusPointFrames checks tracking handles the case when
// maximum score is 0.
TEST(SceneCameraMotionAnalyzerTest,
     PopulateFocusPointFramesTrackingHandlesZeroScore) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  camera_motion.mutable_tracking_motion();
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  scene_summary.set_key_frame_max_score(0.0);
  for (int i = 0; i < kNumKeyFrames; ++i) {
    scene_summary.mutable_key_frame_compact_infos(i)->set_score(0.0);
  }
  std::vector<FocusPointFrame> focus_point_frames;
  MP_EXPECT_OK(analyzer.PopulateFocusPointFrames(
      scene_summary, camera_motion, GetDefaultSceneFrameTimestamps(),
      &focus_point_frames));
}

// Checks that PopulateFocusPointFrames skips empty key frames when camera
// motion type is tracking.
TEST(SceneCameraMotionAnalyzerTest,
     PopulateFocusPointFramesTrackingSkipsEmptyKeyFrames) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  camera_motion.mutable_tracking_motion();
  SceneKeyFrameCropSummary scene_summary;
  scene_summary.set_scene_frame_width(kSceneFrameWidth);
  scene_summary.set_scene_frame_height(kSceneFrameHeight);
  scene_summary.set_num_key_frames(2);

  // Sets first key frame to be empty and second frame to be normal.
  const float center_x = 25.0f, center_y = 25.0f;
  auto* first_frame_compact_info = scene_summary.add_key_frame_compact_infos();
  first_frame_compact_info->set_center_x(-1.0);
  auto* second_frame_compact_info = scene_summary.add_key_frame_compact_infos();
  second_frame_compact_info->set_center_x(center_x);
  second_frame_compact_info->set_center_y(center_y);
  second_frame_compact_info->set_score(1.0);
  scene_summary.set_key_frame_center_min_x(center_x);
  scene_summary.set_key_frame_center_max_x(center_x);
  scene_summary.set_key_frame_center_min_y(center_y);
  scene_summary.set_key_frame_center_max_y(center_y);
  scene_summary.set_key_frame_min_score(1.0);
  scene_summary.set_key_frame_max_score(1.0);

  // Aligns timestamps of scene frames with key frames.
  scene_summary.mutable_key_frame_compact_infos(0)->set_timestamp_ms(10);
  scene_summary.mutable_key_frame_compact_infos(1)->set_timestamp_ms(20);
  std::vector<int64> scene_frame_timestamps = {10, 20};

  std::vector<FocusPointFrame> focus_point_frames;
  MP_EXPECT_OK(analyzer.PopulateFocusPointFrames(scene_summary, camera_motion,
                                                 scene_frame_timestamps,
                                                 &focus_point_frames));

  // Both scene frames should have focus point frames based on the second key
  // frame since the first one is empty and not used.
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(focus_point_frames[i].point_size(), 1);
    const auto& point = focus_point_frames[i].point(0);
    EXPECT_FLOAT_EQ(point.norm_point_x(), center_x / kSceneFrameWidth);
    EXPECT_FLOAT_EQ(point.norm_point_y(), center_y / kSceneFrameHeight);
    EXPECT_FLOAT_EQ(point.weight(), options.maximum_salient_point_weight());
  }
}

// Checks that PopulateFocusPointFrames properly sets FocusPointFrames when
// camera motion type is tracking, piecewise-linearly interpolating key frame
// centers and scores, and scaling scores so that maximum weight is equal to
// maximum_salient_point_weight.
TEST(SceneCameraMotionAnalyzerTest,
     PopulateFocusPointFramesTrackingTracksKeyFrames) {
  SceneCameraMotionAnalyzerOptions options;
  TestableSceneCameraMotionAnalyzer analyzer(options);
  SceneCameraMotion camera_motion;
  camera_motion.mutable_tracking_motion();
  auto scene_summary = GetDefaultSceneKeyFrameCropSummary();
  const std::vector<float> centers_x = {14.0, 5.0, 40.0, 70.0, 30.0};
  const std::vector<float> centers_y = {60.0, 50.0, 80.0, 0.0, 20.0};
  const std::vector<float> scores = {0.1, 1.0, 2.0, 0.6, 0.9};
  scene_summary.set_key_frame_min_score(0.1);
  scene_summary.set_key_frame_max_score(2.0);
  for (int i = 0; i < kNumKeyFrames; ++i) {
    auto* compact_info = scene_summary.mutable_key_frame_compact_infos(i);
    compact_info->set_center_x(centers_x[i]);
    compact_info->set_center_y(centers_y[i]);
    compact_info->set_score(scores[i]);
  }

  // Get reference scene frame results from csv file.
  const std::string scene_frame_results_file_path =
      mediapipe::file::JoinPath("./", kCameraTrackingSceneFrameResultsFile);
  std::string csv_file_content;
  MP_ASSERT_OK(mediapipe::file::GetContents(scene_frame_results_file_path,
                                            &csv_file_content));
  std::vector<std::string> lines = absl::StrSplit(csv_file_content, '\n');
  std::vector<std::string> records;
  for (const auto& line : lines) {
    std::vector<std::string> r = absl::StrSplit(line, ',');
    records.insert(records.end(), r.begin(), r.end());
  }
  CHECK_EQ(records.size(), kNumSceneFrames * 3 + 1);

  std::vector<FocusPointFrame> focus_point_frames;
  MP_EXPECT_OK(analyzer.PopulateFocusPointFrames(
      scene_summary, camera_motion, GetDefaultSceneFrameTimestamps(),
      &focus_point_frames));

  float max_weight = 0.0;
  const float tolerance = 1e-4;
  for (int i = 0; i < kNumSceneFrames; ++i) {
    EXPECT_EQ(focus_point_frames[i].point_size(), 1);
    const auto& point = focus_point_frames[i].point(0);
    const float expected_x = std::stof(records[i * 3]);
    const float expected_y = std::stof(records[i * 3 + 1]);
    const float expected_weight = std::stof(records[i * 3 + 2]);
    EXPECT_LE(std::fabs(point.norm_point_x() - expected_x), tolerance);
    EXPECT_LE(std::fabs(point.norm_point_y() - expected_y), tolerance);
    EXPECT_LE(std::fabs(point.weight() - expected_weight), tolerance);
    max_weight = std::max(max_weight, point.weight());
  }
  EXPECT_LE(std::fabs(max_weight - options.maximum_salient_point_weight()),
            tolerance);
}

// Checks that AnalyzeSceneAndPopulateFocusPointFrames analyzes scene and
// populates focus point frames.
TEST(SceneCameraMotionAnalyzerTest, AnalyzeSceneAndPopulateFocusPointFrames) {
  SceneCameraMotionAnalyzerOptions options;
  SceneCameraMotionAnalyzer analyzer(options);
  SceneKeyFrameCropSummary scene_summary;
  std::vector<FocusPointFrame> focus_point_frames;

  MP_EXPECT_OK(analyzer.AnalyzeSceneAndPopulateFocusPointFrames(
      GetDefaultKeyFrameCropOptions(), GetDefaultKeyFrameCropResults(),
      kSceneFrameWidth, kSceneFrameHeight, GetDefaultSceneFrameTimestamps(),
      false, &scene_summary, &focus_point_frames));
  EXPECT_EQ(scene_summary.num_key_frames(), kNumKeyFrames);
  EXPECT_EQ(focus_point_frames.size(), kNumSceneFrames);
}

// Checks that AnalyzeSceneAndPopulateFocusPointFrames optionally returns
// scene camera motion.
TEST(SceneCameraMotionAnalyzerTest,
     AnalyzeSceneAndPopulateFocusPointFramesReturnsSceneCameraMotion) {
  SceneCameraMotionAnalyzerOptions options;
  SceneCameraMotionAnalyzer analyzer(options);
  SceneKeyFrameCropSummary scene_summary;
  std::vector<FocusPointFrame> focus_point_frames;
  SceneCameraMotion scene_camera_motion;

  MP_EXPECT_OK(analyzer.AnalyzeSceneAndPopulateFocusPointFrames(
      GetDefaultKeyFrameCropOptions(), GetDefaultKeyFrameCropResults(),
      kSceneFrameWidth, kSceneFrameHeight, GetDefaultSceneFrameTimestamps(),
      false, &scene_summary, &focus_point_frames, &scene_camera_motion));
  EXPECT_TRUE(scene_camera_motion.has_steady_motion());
}

}  // namespace autoflip
}  // namespace mediapipe
