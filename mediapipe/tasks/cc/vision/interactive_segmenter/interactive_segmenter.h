/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_VISION_INTERACTIVE_SEGMENTER_INTERACTIVE_SEGMENTER_H_
#define MEDIAPIPE_TASKS_CC_VISION_INTERACTIVE_SEGMENTER_INTERACTIVE_SEGMENTER_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/components/containers/keypoint.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter_result.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace interactive_segmenter {

// The options for configuring a mediapipe interactive segmenter task.
struct InteractiveSegmenterOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // Whether to output confidence masks.
  bool output_confidence_masks = true;

  // Whether to output category mask.
  bool output_category_mask = false;
};

// The Region-Of-Interest (ROI) to interact with.
struct RegionOfInterest {
  enum class Format {
    kUnspecified = 0,  // Format not specified.
    kKeyPoint = 1,     // Using keypoint to represent ROI.
    kScribble = 2,     // Using scribble to represent ROI.
  };

  // Specifies the format used to specify the region-of-interest. Note that
  // using `UNSPECIFIED` is invalid and will lead to an `InvalidArgument` status
  // being returned.
  Format format = Format::kUnspecified;

  // Represents the ROI in keypoint format, this should be non-nullopt if
  // `format` is `kKeyPoint`.
  std::optional<components::containers::NormalizedKeypoint> keypoint;

  // Represents the ROI in scribble format, this should be non-nullopt if
  // `format` is `kScribble`.
  std::optional<std::vector<components::containers::NormalizedKeypoint>>
      scribble;
};

// Performs interactive segmentation on images.
//
// Users can represent user interaction through `RegionOfInterest`, which gives
// a hint to InteractiveSegmenter to perform segmentation focusing on the given
// region of interest.
//
// The API expects a TFLite model with mandatory TFLite Model Metadata.
//
// Input tensor:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    - image input of size `[batch x height x width x channels]`.
//    - batch inference is not supported (`batch` is required to be 1).
//    - RGB inputs is supported (`channels` is required to be 3).
//    - if type is kTfLiteFloat32, NormalizationOptions are required to be
//      attached to the metadata for input normalization.
// Output ImageSegmenterResult:
//    Provides optional confidence masks if `output_confidence_masks` is set
//    true, and an optional category mask if `output_category_mask` is set
//    true. At least one of `output_confidence_masks` and `output_category_mask`
//    must be set to true.
class InteractiveSegmenter : tasks::vision::core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates an InteractiveSegmenter from the provided options. A non-default
  // OpResolver can be specified in the BaseOptions of
  // InteractiveSegmenterOptions, to support custom Ops of the segmentation
  // model.
  static absl::StatusOr<std::unique_ptr<InteractiveSegmenter>> Create(
      std::unique_ptr<InteractiveSegmenterOptions> options);

  // Performs image segmentation on the provided single image.
  //
  // The image can be of any size with format RGB.
  //
  // The `roi` parameter is used to represent user's region of interest for
  // segmentation.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing segmentation, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  absl::StatusOr<image_segmenter::ImageSegmenterResult> Segment(
      mediapipe::Image image, const RegionOfInterest& roi,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Shuts down the InteractiveSegmenter when all works are done.
  absl::Status Close() { return runner_->Close(); }

 private:
  bool output_confidence_masks_;
  bool output_category_mask_;
};

}  // namespace interactive_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_INTERACTIVE_SEGMENTER_INTERACTIVE_SEGMENTER_H_
