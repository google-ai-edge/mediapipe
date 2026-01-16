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

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_SCENE_CAMERA_MOTION_ANALYZER_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_SCENE_CAMERA_MOTION_ANALYZER_H_

#include <cstdint>
#include <vector>

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/cropping.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/focus_point.pb.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

// This class does the following in order:
// - Aggregates a key frame results to get a SceneKeyFrameCropSummary,
// - Determines the SceneCameraMotion for the scene, and then
// - Populates FocusPointFrames to be used as input for the retargeter.
//
// Upstream inputs:
// - std::vector<KeyFrameCropInfo> key_frame_crop_infos.
// - KeyFrameCropOptions key_frame_crop_options.
// - std::vector<KeyFrameCropResult> key_frame_crop_results.
// - int scene_frame_width, scene_frame_height.
// - std::vector<int64> scene_frame_timestamps.
//
// Example usage:
//   SceneCameraMotionAnalyzerOptions options;
//   SceneCameraMotionAnalyzer analyzer(options);
//   SceneKeyFrameCropSummary scene_summary;
//   std::vector<FocusPointFrame> focus_point_frames;
//   ABSL_CHECK_OK(analyzer.AnalyzeScenePopulateFocusPointFrames(
//       key_frame_crop_infos, key_frame_crop_options, key_frame_crop_results,
//       scene_frame_width, scene_frame_height, scene_frame_timestamps,
//       &scene_summary, &focus_point_frames));
class SceneCameraMotionAnalyzer {
 public:
  SceneCameraMotionAnalyzer() = delete;

  explicit SceneCameraMotionAnalyzer(const SceneCameraMotionAnalyzerOptions&
                                         scene_camera_motion_analyzer_options)
      : options_(scene_camera_motion_analyzer_options),
        time_since_last_salient_region_us_(0),
        has_solid_color_background_(false),
        total_scene_frames_(0) {}

  ~SceneCameraMotionAnalyzer() {}

  // Aggregates information from KeyFrameInfos and KeyFrameCropResults into
  // SceneKeyFrameCropSummary, and populates FocusPointFrames given scene
  // frame timestamps. Optionally returns SceneCameraMotion.
  absl::Status AnalyzeSceneAndPopulateFocusPointFrames(
      const KeyFrameCropOptions& key_frame_crop_options,
      const std::vector<KeyFrameCropResult>& key_frame_crop_results,
      const int scene_frame_width, const int scene_frame_height,
      const std::vector<int64_t>& scene_frame_timestamps,
      const bool has_solid_color_background,
      SceneKeyFrameCropSummary* scene_summary,
      std::vector<FocusPointFrame>* focus_point_frames,
      SceneCameraMotion* scene_camera_motion = nullptr);

 protected:
  // Decides SceneCameraMotion based on SceneKeyFrameCropSummary. Updates the
  // crop window in SceneKeyFrameCropSummary in the case of steady motion.
  absl::Status DecideCameraMotionType(
      const KeyFrameCropOptions& key_frame_crop_options,
      const double scene_span_sec, const int64_t end_time_us,
      SceneKeyFrameCropSummary* scene_summary,
      SceneCameraMotion* scene_camera_motion) const;

  // Populates the FocusPointFrames for each scene frame based on
  // SceneKeyFrameCropSummary, SceneCameraMotion, and scene frame timestamps.
  absl::Status PopulateFocusPointFrames(
      const SceneKeyFrameCropSummary& scene_summary,
      const SceneCameraMotion& scene_camera_motion,
      const std::vector<int64_t>& scene_frame_timestamps,
      std::vector<FocusPointFrame>* focus_point_frames) const;

 private:
  // Decides the look-at region when camera is steady.
  absl::Status DecideSteadyLookAtRegion(
      const KeyFrameCropOptions& key_frame_crop_options,
      SceneKeyFrameCropSummary* scene_summary,
      SceneCameraMotion* scene_camera_motion) const;

  // Types of FocusPointFrames: number and placement of FocusPoint's vary.
  enum FocusPointFrameType {
    TOPMOST_AND_BOTTOMMOST = 1,  // (center_x, 0) and (center_x, frame_height)
    LEFTMOST_AND_RIGHTMOST = 2,  // (0, center_y) and (frame_width, center_y)
    CENTER = 3,                  // (center_x, center_y)
  };

  // Adds FocusPoint(s) to given FocusPointFrame given center location,
  // frame size, FocusPointFrameType, weight, and bound.
  absl::Status AddFocusPointsFromCenterTypeAndWeight(
      const float center_x, const float center_y, const int frame_width,
      const int frame_height, const FocusPointFrameType type,
      const float weight, const float bound,
      FocusPointFrame* focus_point_frame) const;

  // Populates the FocusPointFrames for each scene frame based on
  // SceneKeyFrameCropSummary and scene frame timestamps in the case where
  // camera is tracking the crop regions.
  absl::Status PopulateFocusPointFramesForTracking(
      const SceneKeyFrameCropSummary& scene_summary,
      const FocusPointFrameType focus_point_frame_type,
      const std::vector<int64_t>& scene_frame_timestamps,
      std::vector<FocusPointFrame>* focus_point_frames) const;

  // Decide to use steady motion.
  absl::Status ToUseSteadyMotion(const float look_at_center_x,
                                 const float look_at_center_y,
                                 const int crop_window_width,
                                 const int crop_window_height,
                                 SceneKeyFrameCropSummary* scene_summary,
                                 SceneCameraMotion* scene_camera_motion) const;

  // Decide to use sweeping motion.
  absl::Status ToUseSweepingMotion(
      const float start_x, const float start_y, const float end_x,
      const float end_y, const int crop_window_width,
      const int crop_window_height, const double time_duration_in_sec,
      SceneKeyFrameCropSummary* scene_summary,
      SceneCameraMotion* scene_camera_motion) const;

  // Scene camera motion analyzer options.
  SceneCameraMotionAnalyzerOptions options_;

  // Last position
  SceneCameraMotion last_scene_with_salient_region_;
  int64_t time_since_last_salient_region_us_;

  // Scene has solid color background.
  bool has_solid_color_background_;

  // Total number of frames for this scene.
  int total_scene_frames_;
};

}  // namespace autoflip
}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_SCENE_CAMERA_MOTION_ANALYZER_H_
