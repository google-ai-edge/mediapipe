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

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_SCENE_CROPPER_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_SCENE_CROPPER_H_

#include <memory>
#include <vector>

#include "mediapipe/examples/desktop/autoflip/quality/cropping.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/focus_point.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

// This class is a thin wrapper around the Retargeter class to crop a collection
// of scene frames given SceneKeyFrameCropSummary and their FocusPointFrames.
//
// Upstream inputs:
// - SceneKeyFrameCropSummary scene_summary.
// - std::vector<FocusPointFrame> focus_point_frames.
// - std::vector<FocusPointFrame> prior_focus_point_frames.
// - std::vector<cv::Mat> scene_frames;
//
// Example usage:
//   SceneCropperOptions scene_cropper_options;
//   SceneCropper scene_cropper(scene_cropper_options);
//   std::vector<cv::Mat> cropped_frames;
//   CHECK_OK(scene_cropper.CropFrames(
//       scene_summary, scene_frames, focus_point_frames,
//       prior_focus_point_frames, &cropped_frames));
class SceneCropper {
 public:
  SceneCropper() {}
  ~SceneCropper() {}

  // Computes transformation matrix given SceneKeyFrameCropSummary,
  // FocusPointFrames, and any prior FocusPointFrames (to ensure smoothness when
  // there was no actual scene change). Optionally crops the input frames based
  // on the transform matrix if |cropped_frames| is not nullptr and
  // |scene_frames_or_empty| isn't empty.
  // TODO: split this function into two separate functions.
  ::mediapipe::Status CropFrames(
      const SceneKeyFrameCropSummary& scene_summary, const int num_scene_frames,
      const std::vector<cv::Mat>& scene_frames_or_empty,
      const std::vector<FocusPointFrame>& focus_point_frames,
      const std::vector<FocusPointFrame>& prior_focus_point_frames,
      int top_static_border_size, int bottom_static_border_size,
      std::vector<cv::Rect>* all_scene_frame_xforms,
      std::vector<cv::Mat>* cropped_frames) const;
};

}  // namespace autoflip
}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_SCENE_CROPPER_H_
