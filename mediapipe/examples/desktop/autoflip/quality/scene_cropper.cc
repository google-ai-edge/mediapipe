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

#include "mediapipe/examples/desktop/autoflip/quality/scene_cropper.h"

#include "absl/memory/memory.h"
#include "mediapipe/examples/desktop/autoflip/quality/polynomial_regression_path_solver.h"
#include "mediapipe/examples/desktop/autoflip/quality/utils.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

::mediapipe::Status SceneCropper::CropFrames(
    const SceneKeyFrameCropSummary& scene_summary, const int num_scene_frames,
    const std::vector<cv::Mat>& scene_frames_or_empty,
    const std::vector<FocusPointFrame>& focus_point_frames,
    const std::vector<FocusPointFrame>& prior_focus_point_frames,
    int top_static_border_size, int bottom_static_border_size,
    std::vector<cv::Rect>* crop_from_location,
    std::vector<cv::Mat>* cropped_frames) const {
  RET_CHECK_GT(num_scene_frames, 0) << "No scene frames.";
  RET_CHECK_EQ(focus_point_frames.size(), num_scene_frames)
      << "Wrong size of FocusPointFrames.";

  const int frame_width = scene_summary.scene_frame_width();
  const int frame_height = scene_summary.scene_frame_height();
  const int crop_width = scene_summary.crop_window_width();
  const int crop_height = scene_summary.crop_window_height();
  RET_CHECK_GT(crop_width, 0) << "Crop width is non-positive.";
  RET_CHECK_GT(crop_height, 0) << "Crop height is non-positive.";
  RET_CHECK_LE(crop_width, frame_width) << "Crop width exceeds frame width.";
  RET_CHECK_LE(crop_height, frame_height)
      << "Crop height exceeds frame height.";

  // Computes transforms.
  std::vector<cv::Mat> all_xforms;

  PolynomialRegressionPathSolver solver;
  RET_CHECK_OK(solver.ComputeCameraPath(
      focus_point_frames, prior_focus_point_frames, frame_width, frame_height,
      crop_width, crop_height, &all_xforms));

  const int num_prior = prior_focus_point_frames.size();
  std::vector<cv::Mat> scene_frame_xforms(all_xforms.begin() + num_prior,
                                          all_xforms.end());

  // Convert the matrix from center-aligned to upper-left aligned.
  for (cv::Mat& xform : scene_frame_xforms) {
    cv::Mat affine_opencv = cv::Mat::eye(2, 3, CV_32FC1);
    affine_opencv.at<float>(0, 2) =
        -(xform.at<float>(0, 2) + frame_width / 2 - crop_width / 2);
    affine_opencv.at<float>(1, 2) =
        -(xform.at<float>(1, 2) + frame_height / 2 - crop_height / 2);
    xform = affine_opencv;
  }

  // If no cropped_frames is passed in, return directly.
  if (!cropped_frames) {
    return ::mediapipe::OkStatus();
  }
  RET_CHECK(!scene_frames_or_empty.empty())
      << "If |cropped_frames| != nullptr, scene_frames_or_empty must not be "
         "empty.";
  // Prepares cropped frames.
  cropped_frames->resize(num_scene_frames);
  for (int i = 0; i < num_scene_frames; ++i) {
    (*cropped_frames)[i] = cv::Mat::zeros(crop_height, crop_width,
                                          scene_frames_or_empty[i].type());
  }

  // Store the "crop from" location on the input frame for use with an external
  // renderer.
  for (int i = 0; i < num_scene_frames; i++) {
    const int left = scene_frame_xforms[i].at<float>(0, 2);
    const int right = left + crop_width;
    const int top = top_static_border_size;
    const int bottom =
        top_static_border_size +
        (crop_height - top_static_border_size - bottom_static_border_size);
    crop_from_location->push_back(
        cv::Rect(left, top, right - left, bottom - top));
  }

  return AffineRetarget(cv::Size(crop_width, crop_height),
                        scene_frames_or_empty, scene_frame_xforms,
                        cropped_frames);
}

}  // namespace autoflip
}  // namespace mediapipe
