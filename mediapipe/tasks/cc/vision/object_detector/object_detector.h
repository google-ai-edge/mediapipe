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

#ifndef MEDIAPIPE_TASKS_CC_VISION_OBJECT_DETECTOR_OBJECT_DETECTOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_OBJECT_DETECTOR_OBJECT_DETECTOR_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/components/containers/detection_result.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"

namespace mediapipe {
namespace tasks {
namespace vision {

// Alias the shared DetectionResult struct as result typo.
using ObjectDetectorResult =
    ::mediapipe::tasks::components::containers::DetectionResult;

// Options related to non-maximum-suppression.
struct NonMaxSuppressionOptions {
  // Whether to use multiclass non-max-suppression. That is, each category
  // processes non-max-suppression separately.
  bool multiclass_nms = false;

  // Overlapping threshold for non-maximum-suppression. Only used for
  // models without built-in non-maximum-suppression, i.e., models that don't
  // use the Detection_Postprocess TFLite Op
  float min_suppression_threshold = 0.3f;
};

// The options for configuring a mediapipe object detector task.
struct ObjectDetectorOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the TfLite
  // model file with metadata, accelerator options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // Object detector has three running modes:
  // 1) The image mode for detecting objects on single image inputs.
  // 2) The video mode for detecting objects on the decoded frames of a video.
  // 3) The live stream mode for detecting objects on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the detection results asynchronously.
  core::RunningMode running_mode = core::RunningMode::IMAGE;

  // The locale to use for display names specified through the TFLite Model
  // Metadata, if any. Defaults to English.
  std::string display_names_locale = "en";

  // The maximum number of top-scored detection results to return. If < 0, all
  // available results will be returned. If 0, an invalid argument error is
  // returned. Note that models may intrinsically be limited to returning a
  // maximum number of results N: if the provided value here is above N, only N
  // results will be returned.
  int max_results = -1;

  // Score threshold to override the one provided in the model metadata (if
  // any). Detection results with a score below this value are rejected.
  float score_threshold = 0.0f;

  // The allowlist of category names. If non-empty, detection results whose
  // category name is not in this set will be filtered out. Duplicate or unknown
  // category names are ignored. Mutually exclusive with category_denylist.
  std::vector<std::string> category_allowlist = {};

  // The denylist of category names. If non-empty, detection results whose
  // category name is in this set will be filtered out. Duplicate or unknown
  // category names are ignored. Mutually exclusive with category_allowlist.
  std::vector<std::string> category_denylist = {};

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  std::function<void(absl::StatusOr<ObjectDetectorResult>, const Image&,
                     int64_t)>
      result_callback = nullptr;

  NonMaxSuppressionOptions non_max_suppression_options;
};

// Performs object detection on single images, video frames, or live stream.
//
// The API expects a TFLite model with mandatory TFLite Model Metadata.
//
// Input tensor:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    - image input of size `[batch x height x width x channels]`.
//    - batch inference is not supported (`batch` is required to be 1).
//    - only RGB inputs are supported (`channels` is required to be 3).
//    - if type is kTfLiteFloat32, NormalizationOptions are required to be
//      attached to the metadata for input normalization.
// Output tensors could be 2 output tensors or 4 output tensors.
// The 2 output tensors must represent locations and scores, respectively.
//  (kTfLiteFloat32)
//   - locations tensor of size `[num_results x num_coords]`. The num_coords is
//   the number of coordinates a location result represent. Usually in the
//   form: [4 + 2 * keypoint_num], where 4 location values encode the bounding
//   box (y_center, x_center, height, width) and the additional keypoints are in
//   (y, x) order.
//  (kTfLiteFloat32)
//   - scores tensor of size `[num_results x num_classes]`. The values of a
//   result represent the classification probability belonging to the class at
//   the index, which is denoted in the label file of corresponding tensor
//   metadata in the model file.
// The 4 output tensors must come from `DetectionPostProcess` op, i.e:
//  (kTfLiteFloat32)
//   - locations tensor of size `[num_results x 4]`, the inner array
//     representing bounding boxes in the form [top, left, right, bottom].
//   - BoundingBoxProperties are required to be attached to the metadata
//     and must specify type=BOUNDARIES and coordinate_type=RATIO.
//  (kTfLiteFloat32)
//   - classes tensor of size `[num_results]`, each value representing the
//     integer index of a class.
//   - optional (but recommended) label map(s) can be attached as
//     AssociatedFile-s with type TENSOR_VALUE_LABELS, containing one label per
//     line. The first such AssociatedFile (if any) is used to fill the
//     `class_name` field of the results. The `display_name` field is filled
//     from the AssociatedFile (if any) whose locale matches the
//     `display_names_locale` field of the `ObjectDetectorOptions` used at
//     creation time ("en" by default, i.e. English). If none of these are
//     available, only the `index` field of the results will be filled.
//  (kTfLiteFloat32)
//   - scores tensor of size `[num_results]`, each value representing the score
//     of the detected object.
//   - optional score calibration can be attached using ScoreCalibrationOptions
//     and an AssociatedFile with type TENSOR_AXIS_SCORE_CALIBRATION. See
//     metadata_schema.fbs [1] for more details.
//  (kTfLiteFloat32)
//   - integer num_results as a tensor of size `[1]`
//
// An example of such model can be found at:
// https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/metadata/1
//
// [1]:
// https://github.com/google/mediapipe/blob/6cdc6443b6a7ed662744e2a2ce2d58d9c83e6d6f/mediapipe/tasks/metadata/metadata_schema.fbs#L456
class ObjectDetector : public tasks::vision::core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates an ObjectDetector from an ObjectDetectorOptions to process image
  // data or streaming data. Object detector can be created with one of the
  // following three running modes:
  // 1) Image mode for detecting objects on single image inputs.
  //    Users provide mediapipe::Image to the `Detect` method, and will
  //    receive the detection results as the return value.
  // 2) Video mode for detecting objects on the decoded frames of a video.
  // 3) Live stream mode for detecting objects on the live stream of the input
  //    data, such as from camera. Users call `DetectAsync` to push the image
  //    data into the ObjectDetector, the detection results along with the input
  //    timestamp and the image that object detector runs on will be available
  //    in the result callback when the object detector finishes the work.
  static absl::StatusOr<std::unique_ptr<ObjectDetector>> Create(
      std::unique_ptr<ObjectDetectorOptions> options);

  // Performs object detection on the provided single image.
  // Only use this method when the ObjectDetector is created with the image
  // running mode.
  //
  // The image can be of any size with format RGB or RGBA.
  // TODO: Describes how the input image will be preprocessed
  // after the yuv support is implemented.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  //
  // For CPU images, the returned bounding boxes are expressed in the
  // unrotated input frame of reference coordinates system, i.e. in `[0,
  // image_width) x [0, image_height)`, which are the dimensions of the
  // underlying image data.
  // TODO: Describes the output bounding boxes for gpu input
  // images after enabling the gpu support in MediaPipe Tasks.
  absl::StatusOr<ObjectDetectorResult> Detect(
      mediapipe::Image image,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Performs object detection on the provided video frame.
  // Only use this method when the ObjectDetector is created with the video
  // running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide the video frame's timestamp (in milliseconds). The input timestamps
  // must be monotonically increasing.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  //
  // For CPU images, the returned bounding boxes are expressed in the
  // unrotated input frame of reference coordinates system, i.e. in `[0,
  // image_width) x [0, image_height)`, which are the dimensions of the
  // underlying image data.
  absl::StatusOr<ObjectDetectorResult> DetectForVideo(
      mediapipe::Image image, int64_t timestamp_ms,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Sends live image data to perform object detection, and the results will be
  // available via the "result_callback" provided in the ObjectDetectorOptions.
  // Only use this method when the ObjectDetector is created with the live
  // stream running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide a timestamp (in milliseconds) to indicate when the input image is
  // sent to the object detector. The input timestamps must be monotonically
  // increasing.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  //
  // The "result_callback" provides
  //   - A vector of detections, each has a bounding box that is expressed in
  //     the unrotated input frame of reference coordinates system, i.e. in `[0,
  //     image_width) x [0, image_height)`, which are the dimensions of the
  //     underlying image data.
  //   - The const reference to the corresponding input image that the object
  //     detector runs on. Note that the const reference to the image will no
  //     longer be valid when the callback returns. To access the image data
  //     outside of the callback, callers need to make a copy of the image.
  //   - The input timestamp in milliseconds.
  absl::Status DetectAsync(mediapipe::Image image, int64_t timestamp_ms,
                           std::optional<core::ImageProcessingOptions>
                               image_processing_options = std::nullopt);

  // Shuts down the ObjectDetector when all works are done.
  absl::Status Close() { return runner_->Close(); }
};

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_OBJECT_DETECTOR_OBJECT_DETECTOR_H_
