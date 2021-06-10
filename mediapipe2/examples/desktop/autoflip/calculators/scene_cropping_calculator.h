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

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_CALCULATORS_SCENE_CROPPING_CALCULATOR_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_CALCULATORS_SCENE_CROPPING_CALCULATOR_H_

#include <memory>
#include <vector>

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/scene_cropping_calculator.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/cropping.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/focus_point.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/frame_crop_region_computer.h"
#include "mediapipe/examples/desktop/autoflip/quality/padding_effect_generator.h"
#include "mediapipe/examples/desktop/autoflip/quality/piecewise_linear_function.h"
#include "mediapipe/examples/desktop/autoflip/quality/polynomial_regression_path_solver.h"
#include "mediapipe/examples/desktop/autoflip/quality/scene_camera_motion_analyzer.h"
#include "mediapipe/examples/desktop/autoflip/quality/scene_cropper.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {
// This calculator crops video scenes to target size, which can be of any aspect
// ratio. The calculator supports both "landscape -> portrait", and "portrait ->
// landscape" use cases. The two use cases are automatically determined by
// comparing the input and output frame's aspect ratios internally.
//
// The target (i.e. output) frame's dimension can be specified through the
// target_width(height) fields in the options. Both this target dimension and
// the input dimension should be even. If either keep_original_height or
// keep_original_width is set to true, the corresponding target dimension will
// only be used to compute the aspect ratio (as opposed to setting the actual
// dimension) of the output. If the output frame thus computed has an odd
// size, it will be rounded down to an even number.
//
// The calculator takes shot boundary signals to identify shot boundaries, and
// crops each scene independently. The cropping decisions are made based on
// detection features, which are a collection of focus regions detected from
// different signals, and then fused together by a SignalFusingCalculator. To
// add a new type of focus signals, it should be added in the input of the
// SignalFusingCalculator, which can take an arbitrary number of input streams.
//
// If after attempting to cover focus regions based on the cropping decisions
// made, the retained frame region's aspect ratio is still different from the
// target aspect ratio, padding will be applied. In this case, a seamless
// padding with a solid color would be preferred wherever possible, given
// information from the input static features; otherwise, a simple padding with
// centered foreground on blurred background will be applied.
//
// The main complexity of this calculator lies in stabilizing crop regions over
// the scene using a Retargeter, which solves linear programming problems
// through a L1 path solver (default) or least squares problems through a L2
// path solver.

// Input streams:
// - required tag VIDEO_FRAMES (type ImageFrame):
//     Original scene frames to be cropped.
// - required tag DETECTION_FEATURES (type DetectionSet):
//     Detected features on the key frames.
// - optional tag STATIC_FEATURES (type StaticFeatures):
//     Detected features on the key frames.
// - required tag SHOT_BOUNDARIES (type bool):
//     Indicators for shot boundaries (output of shot boundary detection).
// - optional tag KEY_FRAMES (type ImageFrame):
//     Key frames on which features are detected. This is only used to set the
//     detection features frame size.  Alternatively, set
//     video_feature_width/video_features_height within the options proto to
//     define this value.  When neither is set, the features frame size is
//     assumed to be the original scene frame size.
//
// Output streams:
// - required tag CROPPED_FRAMES (type ImageFrame):
//     Cropped frames at target size and original frame rate.
// - optional tag KEY_FRAME_CROP_REGION_VIZ_FRAMES (type ImageFrame):
//     Debug visualization frames at original frame size and frame rate. Draws
//     the required (yellow) and non-required (cyan) detection features and the
//     key frame crop regions (green).
// - optional tag SALIENT_POINT_FRAME_VIZ_FRAMES (type ImageFrame):
//     Debug visualization frames at original frame size and frame rate. Draws
//     the focus points and the scene crop window (red).
// - optional tag CROPPING_SUMMARY (type VideoCroppingSummary):
//     Debug summary information for the video. Only generates one packet when
//     calculator closes.
// - optional tag EXTERNAL_RENDERING_PER_FRAME (type ExternalRenderFrame)
//     Provides a per-frame message that can be used to render autoflip using an
//     external renderer.
// - optional tag EXTERNAL_RENDERING_FULL_VID (type Vector<ExternalRenderFrame>)
//     Provides an end-stream message that can be used to render autoflip using
//     an external renderer.
//
// Example config:
// node {
//   calculator: "SceneCroppingCalculator"
//   input_stream: "VIDEO_FRAMES:camera_frames_org"
//   input_stream: "KEY_FRAMES:down_sampled_frames"
//   input_stream: "DETECTION_FEATURES:focus_regions"
//   input_stream: "STATIC_FEATURES:border_features"
//   input_stream: "SHOT_BOUNDARIES:shot_boundary_frames"
//   output_stream: "CROPPED_FRAMES:cropped_frames"
//   options: {
//     [mediapipe.SceneCroppingCalculatorOptions.ext]: {
//       target_width: 720
//       target_height: 1124
//       target_size_type: USE_TARGET_DIMENSION
//     }
//   }
// }
// Note that only the target size is required in the options, and all other
// fields are optional with default settings.
class SceneCroppingCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  // Validates calculator options and initializes SceneCameraMotionAnalyzer and
  // SceneCropper.
  absl::Status Open(CalculatorContext* cc) override;

  // Buffers each scene frame and its timestamp. Packs and stores KeyFrameInfo
  // for key frames (a.k.a. frames with detection features). When a shot
  // boundary is encountered or when the buffer is full, calls ProcessScene()
  // to process the scene at once, and clears buffers.
  absl::Status Process(CalculatorContext* cc) override;

  // Calls ProcessScene() on remaining buffered frames. Optionally outputs a
  // VideoCroppingSummary if the output stream CROPPING_SUMMARY is present.
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  // Removes any static borders from the scene frames before cropping. The
  // arguments |top_border_size| and |bottom_border_size| report the size of the
  // removed borders.
  absl::Status RemoveStaticBorders(CalculatorContext* cc, int* top_border_size,
                                   int* bottom_border_size);

  // Sets up autoflip after first frame is received and input size is known.
  absl::Status InitializeSceneCroppingCalculator(
      mediapipe::CalculatorContext* cc);
  // Initializes a FrameCropRegionComputer given input and target frame sizes.
  absl::Status InitializeFrameCropRegionComputer();

  // Processes a scene using buffered scene frames and KeyFrameInfos:
  // 1. Computes key frame crop regions using a FrameCropRegionComputer.
  // 2. Analyzes scene camera motion and generates FocusPointFrames using a
  //    SceneCameraMotionAnalyzer.
  // 3. Crops scene frames using a SceneCropper (wrapper around Retargeter).
  // 4. Formats and outputs cropped frames .
  // 5. Caches prior FocusPointFrames if this is not the end of a scene (due
  //    to force flush).
  // 6. Optionally outputs visualization frames.
  // 7. Optionally updates cropping summary.
  absl::Status ProcessScene(const bool is_end_of_scene, CalculatorContext* cc);

  // Formats and outputs the cropped frames passed in through
  // |cropped_frames_ptr|. Scales them to be at least as big as the target
  // size. If the aspect ratio is different, applies padding. Uses solid
  // background from static features if possible, otherwise uses blurred
  // background. Sets |apply_padding| to true if the scene is padded. Set
  // |cropped_frames_ptr| to nullptr, to bypass the actual output of the
  // cropped frames. This is useful when the calculator is only used for
  // computing the cropping metadata rather than doing the actual cropping
  // operation.
  absl::Status FormatAndOutputCroppedFrames(
      const int crop_width, const int crop_height, const int num_frames,
      std::vector<cv::Rect>* render_to_locations, bool* apply_padding,
      std::vector<cv::Scalar>* padding_colors, float* vertical_fill_percent,
      const std::vector<cv::Mat>* cropped_frames_ptr, CalculatorContext* cc);

  // Draws and outputs visualization frames if those streams are present.
  absl::Status OutputVizFrames(
      const std::vector<KeyFrameCropResult>& key_frame_crop_results,
      const std::vector<FocusPointFrame>& focus_point_frames,
      const std::vector<cv::Rect>& crop_from_locations,
      const int crop_window_width, const int crop_window_height,
      CalculatorContext* cc) const;

  // Filters detections based on USER_HINT under specific flag conditions.
  void FilterKeyFrameInfo();

  // Target frame size and aspect ratio passed in or computed from options.
  int target_width_ = -1;
  int target_height_ = -1;
  double target_aspect_ratio_ = -1.0;

  // Input video frame size and format.
  int frame_width_ = -1;
  int frame_height_ = -1;
  ImageFormat::Format frame_format_ = ImageFormat::UNKNOWN;

  // Key frame size (frame size for detections and border detections).
  int key_frame_width_ = -1;
  int key_frame_height_ = -1;

  // Calculator options.
  SceneCroppingCalculatorOptions options_;

  // Buffered KeyFrameInfos for the current scene (size = number of key
  // frames).
  std::vector<KeyFrameInfo> key_frame_infos_;

  // Buffered frames, timestamps, and indicators for key frames in the current
  // scene (size = number of input video frames).
  // Note: scene_frames_or_empty_ may be empty if the actual cropping
  // operation of frames is turned off, e.g. when
  // |should_perform_frame_cropping_| is false, so rely on
  // scene_frame_timestamps_.size() to query the number of accumulated
  // timestamps rather than scene_frames_or_empty_.size().
  // TODO: all of the following vectors are expected to be the same
  // size. Add to struct and store together in one vector.
  std::vector<cv::Mat> scene_frames_or_empty_;
  std::vector<cv::Mat> raw_scene_frames_or_empty_;
  std::vector<int64> scene_frame_timestamps_;
  std::vector<bool> is_key_frames_;

  // Static border information for the scene.
  int top_border_distance_ = -1;
  int effective_frame_height_ = -1;

  // Stored FocusPointFrames from prior scene when there was no actual scene
  // change (due to forced flush when buffer is full).
  std::vector<FocusPointFrame> prior_focus_point_frames_;
  // Indicates if this scene is a continuation of the last scene (due to
  // forced flush when buffer is full).
  bool continue_last_scene_ = false;

  // KeyFrameCropOptions used by the FrameCropRegionComputer.
  KeyFrameCropOptions key_frame_crop_options_;

  // Object for computing key frame crop regions from detection features.
  std::unique_ptr<FrameCropRegionComputer> frame_crop_region_computer_ =
      nullptr;

  // Object for analyzing scene camera motion from key frame crop regions and
  // generating FocusPointFrames.
  std::unique_ptr<SceneCameraMotionAnalyzer> scene_camera_motion_analyzer_ =
      nullptr;

  // Object for cropping a scene given FocusPointFrames.
  std::unique_ptr<SceneCropper> scene_cropper_ = nullptr;

  // Buffered static features and their timestamps used in padding with solid
  // background color (size = number of frames with static features).
  std::vector<StaticFeatures> static_features_;
  std::vector<int64> static_features_timestamps_;
  bool has_solid_background_ = false;
  // CIELAB yields more natural color transitions than RGB and HSV: RGB tends
  // to produce darker in-between colors and HSV can introduce new hues. See
  // https://howaboutanorange.com/blog/2011/08/10/color_interpolation/ for
  // visual comparisons of color transition in different spaces.
  PiecewiseLinearFunction background_color_l_function_;  // CIELAB - l
  PiecewiseLinearFunction background_color_a_function_;  // CIELAB - a
  PiecewiseLinearFunction background_color_b_function_;  // CIELAB - b

  // Parameters for padding with blurred background passed in from options.
  float background_contrast_ = -1.0;
  int blur_cv_size_ = -1;
  float overlay_opacity_ = -1.0;
  // Object for padding an image to a target aspect ratio.
  std::unique_ptr<PaddingEffectGenerator> padder_ = nullptr;

  // Optional diagnostic summary output emitted in Close().
  std::unique_ptr<VideoCroppingSummary> summary_ = nullptr;

  // Optional list of external rendering messages for each processed frame.
  std::unique_ptr<std::vector<ExternalRenderFrame>> external_render_list_;

  // Determines whether to perform real cropping on input frames. This flag is
  // useful when the user only needs to compute cropping windows, in which
  // case setting this flag to false can avoid buffering as well as cropping
  // frames. This can significantly reduce memory usage and speed up
  // processing. Some debugging visualization inevitably will be disabled
  // because of this flag too.
  bool should_perform_frame_cropping_ = false;
};
}  // namespace autoflip
}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_CALCULATORS_SCENE_CROPPING_CALCULATOR_H_
