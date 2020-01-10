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

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_UTILS_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_UTILS_H_

#include <vector>

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/cropping.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/piecewise_linear_function.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

// Packs detected features and timestamp (ms) into a KeyFrameInfo object. Scales
// features back to the original frame size if features have been detected on a
// different frame size.
::mediapipe::Status PackKeyFrameInfo(const int64 frame_timestamp_ms,
                                     const DetectionSet& detections,
                                     const int original_frame_width,
                                     const int original_frame_height,
                                     const int feature_frame_width,
                                     const int feature_frame_height,
                                     KeyFrameInfo* key_frame_info);

// Sorts required and non-required salient regions given a detection set.
::mediapipe::Status SortDetections(
    const DetectionSet& detections,
    std::vector<SalientRegion>* required_regions,
    std::vector<SalientRegion>* non_required_regions);

// Sets the target crop size in KeyFrameCropOptions based on frame size and
// target aspect ratio so that the target crop size covers the biggest area
// possible in the frame.
::mediapipe::Status SetKeyFrameCropTarget(const int frame_width,
                                          const int frame_height,
                                          const double target_aspect_ratio,
                                          KeyFrameCropOptions* crop_options);

// Aggregates information from KeyFrameInfos and KeyFrameCropResults into
// SceneKeyFrameCropSummary.
::mediapipe::Status AggregateKeyFrameResults(
    const std::vector<KeyFrameInfo>& key_frame_infos,
    const KeyFrameCropOptions& key_frame_crop_options,
    const std::vector<KeyFrameCropResult>& key_frame_crop_results,
    const int scene_frame_width, const int scene_frame_height,
    SceneKeyFrameCropSummary* scene_summary);

// Computes the static top and border size across a scene given a vector of
// StaticFeatures over frames.
::mediapipe::Status ComputeSceneStaticBordersSize(
    const std::vector<StaticFeatures>& static_features, int* top_border_size,
    int* bottom_border_size);

// Finds the solid background colors in a scene from input StaticFeatures.
// Sets has_solid_background to true if the number of frames with solid
// background color exceeds given threshold, i.e.,
// min_fraction_solid_background_color. Builds the background color
// interpolation functions in Lab space using input timestamps.
::mediapipe::Status FindSolidBackgroundColor(
    const std::vector<StaticFeatures>& static_features,
    const std::vector<int64>& static_features_timestamps,
    const double min_fraction_solid_background_color,
    bool* has_solid_background,
    PiecewiseLinearFunction* background_color_l_function,
    PiecewiseLinearFunction* background_color_a_function,
    PiecewiseLinearFunction* background_color_b_function);

// Helpers to scale, clamp, and take union of rectangles. These functions do not
// check for pointers not being null or rectangles being valid.

// Scales a rectangle given horizontal and vertical scaling factors.
template <typename T>
void ScaleRect(const T& original_location, const double scale_x,
               const double scale_y, Rect* scaled_location);

// Converts a normalized rectangle to a rectangle given width and height.
void NormalizedRectToRect(const RectF& normalized_location, const int width,
                          const int height, Rect* location);

// Clamps a rectangle to lie within [x0, y0] and [x1, y1]. Returns true if the
// rectangle has any overlapping with the target window.
::mediapipe::Status ClampRect(const int x0, const int y0, const int x1,
                              const int y1, Rect* location);

// Convenience function to clamp a rectangle to lie within [0, 0] and
// [width, height].
::mediapipe::Status ClampRect(const int width, const int height,
                              Rect* location);

// Enlarges a given rectangle to cover a new rectangle to be added.
void RectUnion(const Rect& rect_to_add, Rect* rect);

// Performs an affine retarget on a list of input images.  Output vector
// cropped_frames must be filled with Mats of the same size as output_size and
// type.
::mediapipe::Status AffineRetarget(
    const cv::Size& output_size, const std::vector<cv::Mat>& frames,
    const std::vector<cv::Mat>& affine_projection,
    std::vector<cv::Mat>* cropped_frames);

}  // namespace autoflip
}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_AUTOFLIP_QUALITY_UTILS_H_
