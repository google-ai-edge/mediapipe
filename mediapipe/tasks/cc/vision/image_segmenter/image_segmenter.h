/* Copyright 2022 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_IMAGE_SEGMENTER_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_IMAGE_SEGMENTER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter_result.h"
#include "tensorflow/lite/kernels/register.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_segmenter {

// The options for configuring a mediapipe image segmenter task.
struct ImageSegmenterOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // Image segmenter has three running modes:
  // 1) The image mode for segmenting image on single image inputs.
  // 2) The video mode for segmenting image on the decoded frames of a video.
  // 3) The live stream mode for segmenting image on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the segmentation results asynchronously.
  core::RunningMode running_mode = core::RunningMode::IMAGE;

  // The locale to use for display names specified through the TFLite Model
  // Metadata, if any. Defaults to English.
  std::string display_names_locale = "en";

  // Whether to output confidence masks.
  bool output_confidence_masks = true;

  // Whether to output category mask.
  bool output_category_mask = false;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  std::function<void(absl::StatusOr<ImageSegmenterResult>, const Image&,
                     int64_t)>
      result_callback = nullptr;
};

// Options for configuring runtime behavior of ImageSegmenter.
struct SegmentationOptions {
  // The width of the output segmentation masks.
  int output_width;

  // The height of the output segmentation masks.
  int output_height;

  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing segmentation, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  std::optional<core::ImageProcessingOptions> image_processing_options;
};

// Performs segmentation on images.
//
// The API expects a TFLite model with mandatory TFLite Model Metadata.
//
// Input tensor:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    - image input of size `[batch x height x width x channels]`.
//    - batch inference is not supported (`batch` is required to be 1).
//    - RGB and greyscale inputs are supported (`channels` is required to be
//      1 or 3).
//    - if type is kTfLiteFloat32, NormalizationOptions are required to be
//      attached to the metadata for input normalization.
// Output ImageSegmenterResult:
//    Provides optional confidence masks if `output_confidence_masks` is set
//    true,  and an optional category mask if `output_category_mask` is set
//    true. At least one of `output_confidence_masks` and `output_category_mask`
//    must be set to true.
// An example of such model can be found at:
// https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2
class ImageSegmenter : tasks::vision::core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates an ImageSegmenter from the provided options. A non-default
  // OpResolver can be specified in the BaseOptions of ImageSegmenterOptions,
  // to support custom Ops of the segmentation model.
  static absl::StatusOr<std::unique_ptr<ImageSegmenter>> Create(
      std::unique_ptr<ImageSegmenterOptions> options);

  // Performs image segmentation on the provided single image.
  // Only use this method when the ImageSegmenter is created with the image
  // running mode.
  //
  // The image can be of any size with format RGB or RGBA.
  //
  // The output size is the same as the input image size.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing segmentation, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  absl::StatusOr<ImageSegmenterResult> Segment(
      mediapipe::Image image,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Performs image segmentation on the provided single image.
  // Only use this method when the ImageSegmenter is created with the image
  // running mode.
  //
  // The image can be of any size with format RGB or RGBA.
  absl::StatusOr<ImageSegmenterResult> Segment(
      mediapipe::Image image, SegmentationOptions segmentation_options);

  // Performs image segmentation on the provided video frame.
  // Only use this method when the ImageSegmenter is created with the video
  // running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide the video frame's timestamp (in milliseconds). The input timestamps
  // must be monotonically increasing.
  //
  // The output size is the same as the input image size.
  //
  // The optional 'image_processing_options' parameter can be used
  // to specify the rotation to apply to the image before performing
  // segmentation, by setting its 'rotation_degrees' field. Note that specifying
  // a region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  absl::StatusOr<ImageSegmenterResult> SegmentForVideo(
      mediapipe::Image image, int64_t timestamp_ms,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Performs image segmentation on the provided video frame.
  // Only use this method when the ImageSegmenter is created with the video
  // running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide the video frame's timestamp (in milliseconds). The input timestamps
  // must be monotonically increasing.
  absl::StatusOr<ImageSegmenterResult> SegmentForVideo(
      mediapipe::Image image, int64_t timestamp_ms,
      SegmentationOptions segmentation_options);

  // Sends live image data to perform image segmentation, and the results will
  // be available via the "result_callback" provided in the
  // ImageSegmenterOptions. Only use this method when the ImageSegmenter is
  // created with the live stream running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide a timestamp (in milliseconds) to indicate when the input image is
  // sent to the image segmenter. The input timestamps must be monotonically
  // increasing.
  //
  // The output size is the same as the input image size.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing segmentation, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  //
  // The "result_callback" provides
  //   - An ImageSegmenterResult.
  //   - The const reference to the corresponding input image that the image
  //     segmentation runs on. Note that the const reference to the image will
  //     no longer be valid when the callback returns. To access the image data
  //     outside of the callback, callers need to make a copy of the image.
  //   - The input timestamp in milliseconds.
  absl::Status SegmentAsync(mediapipe::Image image, int64_t timestamp_ms,
                            std::optional<core::ImageProcessingOptions>
                                image_processing_options = std::nullopt);

  // Sends live image data to perform image segmentation, and the results will
  // be available via the "result_callback" provided in the
  // ImageSegmenterOptions. Only use this method when the ImageSegmenter is
  // created with the live stream running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide a timestamp (in milliseconds) to indicate when the input image is
  // sent to the image segmenter. The input timestamps must be monotonically
  // increasing.
  //
  // The "result_callback" provides
  //   - An ImageSegmenterResult.
  //   - The const reference to the corresponding input image that the image
  //     segmentation runs on. Note that the const reference to the image will
  //     no longer be valid when the callback returns. To access the image data
  //     outside of the callback, callers need to make a copy of the image.
  //   - The input timestamp in milliseconds.
  absl::Status SegmentAsync(mediapipe::Image image, int64_t timestamp_ms,
                            SegmentationOptions segmentation_options);

  // Shuts down the ImageSegmenter when all works are done.
  absl::Status Close() { return runner_->Close(); }

  // Get the category label list of the ImageSegmenter can recognize. For
  // CATEGORY_MASK, the index in the category mask corresponds to the category
  // in the label list. For CONFIDENCE_MASK, the output mask list at index
  // corresponds to the category in the label list.
  //
  // If there is no labelmap provided in the model file, empty label list is
  // returned.
  std::vector<std::string> GetLabels() { return labels_; }

 private:
  std::vector<std::string> labels_;
  bool output_confidence_masks_;
  bool output_category_mask_;

  absl::Status ValidateSegmentationOptions(const SegmentationOptions& options) {
    if (options.output_width <= 0 || options.output_height <= 0) {
      return absl::InvalidArgumentError(
          "Both output_width and output_height must be larger than 0.");
    }
    return absl::OkStatus();
  }
};

}  // namespace image_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_IMAGE_SEGMENTER_H_
