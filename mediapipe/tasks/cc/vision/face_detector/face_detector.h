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

#ifndef MEDIAPIPE_TASKS_CC_VISION_FACE_DETECTOR_FACE_DETECTOR_H_
#define MEDIAPIPE_TASKS_CC_VISION_FACE_DETECTOR_FACE_DETECTOR_H_

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
namespace face_detector {

using FaceDetectorResult =
    ::mediapipe::tasks::components::containers::DetectionResult;

// The options for configuring a mediapipe face detector task.
struct FaceDetectorOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the TfLite
  // model file with metadata, accelerator options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // Face Detector has three running modes:
  // 1) The image mode for detecting faces on single image inputs.
  // 2) The video mode for detecting faces on the decoded frames of a video.
  // 3) The live stream mode for detecting faces on the live stream of input
  // data, such as from camera. In this mode, the "result_callback" below must
  // be specified to receive the detection results asynchronously.
  core::RunningMode running_mode = core::RunningMode::IMAGE;

  // The minimum confidence score for the face detection to be considered
  // successful.
  float min_detection_confidence = 0.5;

  // The minimum non-maximum-suppression threshold for face detection to be
  // considered overlapped.
  float min_suppression_threshold = 0.3;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  std::function<void(absl::StatusOr<FaceDetectorResult>, const Image&,
                     uint64_t)>
      result_callback = nullptr;
};

class FaceDetector : core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates a FaceDetector from a FaceDetectorOptions to process image data
  // or streaming data. Face detector can be created with one of the following
  // three running modes:
  // 1) Image mode for detecting faces on single image inputs. Users
  //    provide mediapipe::Image to the `Detect` method, and will receive the
  //    detected face detection results as the return value.
  // 2) Video mode for detecting faces on the decoded frames of a
  //    video. Users call `DetectForVideo` method, and will receive the detected
  //    face detection results as the return value.
  // 3) Live stream mode for detecting faces on the live stream of the
  //    input data, such as from camera. Users call `DetectAsync` to push the
  //    image data into the FaceDetector, the detected results along with the
  //    input timestamp and the image that face detector runs on will be
  //    available in the result callback when the face detector finishes the
  //    work.
  static absl::StatusOr<std::unique_ptr<FaceDetector>> Create(
      std::unique_ptr<FaceDetectorOptions> options);

  // Performs face detection on the given image.
  // Only use this method when the FaceDetector is created with the image
  // running mode.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by setting
  // its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  //
  // The image can be of any size with format RGB or RGBA.
  // TODO: Describes how the input image will be preprocessed
  // after the yuv support is implemented.
  absl::StatusOr<FaceDetectorResult> Detect(
      Image image,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Performs face detection on the provided video frame.
  // Only use this method when the FaceDetector is created with the video
  // running mode.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by setting
  // its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide the video frame's timestamp (in milliseconds). The input timestamps
  // must be monotonically increasing.
  absl::StatusOr<FaceDetectorResult> DetectForVideo(
      Image image, uint64_t timestamp_ms,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Sends live image data to perform face detection, and the results
  // will be available via the "result_callback" provided in the
  // FaceDetectorOptions. Only use this method when the FaceDetector
  // is created with the live stream running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide a timestamp (in milliseconds) to indicate when the input image is
  // sent to the face detector. The input timestamps must be monotonically
  // increasing.

  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by setting
  // its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  //
  // The "result_callback" provides
  //   - A vector of FaceDetectorResult, each is the detected results
  //     for a input frame.
  //   - The const reference to the corresponding input image that the face
  //     detector runs on. Note that the const reference to the image will no
  //     longer be valid when the callback returns. To access the image data
  //     outside of the callback, callers need to make a copy of the image.
  //   - The input timestamp in milliseconds.
  absl::Status DetectAsync(Image image, uint64_t timestamp_ms,
                           std::optional<core::ImageProcessingOptions>
                               image_processing_options = std::nullopt);

  // Shuts down the FaceDetector when all works are done.
  absl::Status Close() { return runner_->Close(); }
};

}  // namespace face_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_FACE_DETECTOR_FACE_DETECTOR_H_
