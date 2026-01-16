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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKER_H_
#define MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKER_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/tasks/cc/components/processors/classifier_options.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

struct HandLandmarkerOptions {
  // Base options for configuring MediaPipe Tasks library, such as specifying
  // the TfLite model bundle file with metadata, accelerator options, op
  // resolver, etc.
  tasks::core::BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // HandLandmarker has three running modes:
  // 1) The image mode for detecting hand landmarks on single image inputs.
  // 2) The video mode for detecting hand landmarks on the decoded frames of a
  //    video.
  // 3) The live stream mode for detecting hand landmarks on the live stream of
  //    input data, such as from camera. In this mode, the "result_callback"
  //    below must be specified to receive the detection results asynchronously.
  core::RunningMode running_mode = core::RunningMode::IMAGE;

  // The maximum number of hands can be detected by the HandLandmarker.
  int num_hands = 1;

  // The minimum confidence score for the hand detection to be considered
  // successful.
  float min_hand_detection_confidence = 0.5;

  // The minimum confidence score of hand presence score in the hand landmark
  // detection.
  float min_hand_presence_confidence = 0.5;

  // The minimum confidence score for the hand tracking to be considered
  // successful.
  float min_tracking_confidence = 0.5;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  std::function<void(absl::StatusOr<HandLandmarkerResult>, const Image&,
                     int64_t)>
      result_callback = nullptr;
};

// Performs hand landmarks detection on the given image.
//
// TODO  add the link to DevSite.
// This API expects a pre-trained hand landmarker model asset bundle.
//
// Inputs:
//   Image
//     - The image that hand landmarks detection runs on.
//   std::optional<NormalizedRect>
//     - If provided, can be used to specify the rotation to apply to the image
//       before performing hand landmarks detection, by setting its 'rotation'
//       field in radians (e.g. 'M_PI / 2' for a 90° anti-clockwise rotation).
//       Note that specifying a region-of-interest using the 'x_center',
//       'y_center', 'width' and 'height' fields is NOT supported and will
//       result in an invalid argument error being returned.
// Outputs:
//   HandLandmarkerResult
//     - The hand landmarks detection results.
class HandLandmarker : tasks::vision::core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates a HandLandmarker from a HandLandmarkerOptions to process image data
  // or streaming data. Hand landmarker can be created with one of the following
  // three running modes:
  // 1) Image mode for detecting hand landmarks on single image inputs. Users
  //    provide mediapipe::Image to the `Detect` method, and will receive the
  //    detected hand landmarks results as the return value.
  // 2) Video mode for detecting hand landmarks on the decoded frames of a
  //    video. Users call `DetectForVideo` method, and will receive the detected
  //    hand landmarks results as the return value.
  // 3) Live stream mode for detecting hand landmarks on the live stream of the
  //    input data, such as from camera. Users call `DetectAsync` to push the
  //    image data into the HandLandmarker, the detected results along with the
  //    input timestamp and the image that hand landmarker runs on will be
  //    available in the result callback when the hand landmarker finishes the
  //    work.
  static absl::StatusOr<std::unique_ptr<HandLandmarker>> Create(
      std::unique_ptr<HandLandmarkerOptions> options);

  // Performs hand landmarks detection on the given image.
  // Only use this method when the HandLandmarker is created with the image
  // running mode.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by setting
  // its 'rotation_degrees' field. Note that specifying a region-of-interest
  // using the 'region_of_interest' field is NOT supported and will result in an
  // invalid argument error being returned.
  //
  // The image can be of any size with format RGB or RGBA.
  // TODO: Describes how the input image will be preprocessed
  // after the yuv support is implemented.
  absl::StatusOr<HandLandmarkerResult> Detect(
      Image image,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Performs hand landmarks detection on the provided video frame.
  // Only use this method when the HandLandmarker is created with the video
  // running mode.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by setting
  // its 'rotation_degrees' field. Note that specifying a region-of-interest
  // using the 'region_of_interest' field is NOT supported and will result in an
  // invalid argument error being returned.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide the video frame's timestamp (in milliseconds). The input timestamps
  // must be monotonically increasing.
  absl::StatusOr<HandLandmarkerResult> DetectForVideo(
      Image image, int64_t timestamp_ms,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Sends live image data to perform hand landmarks detection, and the results
  // will be available via the "result_callback" provided in the
  // HandLandmarkerOptions. Only use this method when the HandLandmarker
  // is created with the live stream running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide a timestamp (in milliseconds) to indicate when the input image is
  // sent to the hand landmarker. The input timestamps must be monotonically
  // increasing.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by setting
  // its 'rotation_degrees' field. Note that specifying a region-of-interest
  // using the 'region_of_interest' field is NOT supported and will result in an
  // invalid argument error being returned.
  //
  // The "result_callback" provides
  //   - A vector of HandLandmarkerResult, each is the detected results
  //     for a input frame.
  //   - The const reference to the corresponding input image that the hand
  //     landmarker runs on. Note that the const reference to the image will no
  //     longer be valid when the callback returns. To access the image data
  //     outside of the callback, callers need to make a copy of the image.
  //   - The input timestamp in milliseconds.
  absl::Status DetectAsync(Image image, int64_t timestamp_ms,
                           std::optional<core::ImageProcessingOptions>
                               image_processing_options = std::nullopt);

  // Shuts down the HandLandmarker when all works are done.
  absl::Status Close() { return runner_->Close(); }
};

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HAND_LANDMARKER_HAND_LANDMARKER_H_
