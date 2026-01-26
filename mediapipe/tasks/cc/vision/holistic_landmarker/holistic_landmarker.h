/* Copyright 2026 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_LANDMARKER_H_
#define MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_LANDMARKER_H_

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_landmarker_result.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

struct HolisticLandmarkerOptions {
  // Base options for configuring MediaPipe Tasks library, such as specifying
  // the TfLite model bundle file with metadata, accelerator options, op
  // resolver, etc.
  tasks::core::BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // HolisticLandmarker has three running modes:
  // 1) The image mode for detecting holistic landmarks on single image inputs.
  // 2) The video mode for detecting holistic landmarks on the decoded frames of
  //    a video.
  // 3) The live stream mode for detecting holistic landmarks on the live stream
  //    of input data, such as from camera. In this mode, the "result_callback"
  //    below must be specified to receive the detection results asynchronously.
  core::RunningMode running_mode = core::RunningMode::IMAGE;

  // The minimum confidence score for the face detection to be considered
  // successful.
  float min_face_detection_confidence = 0.5;

  // The minimum threshold for the face suppression score in the face detection.
  float min_face_suppression_threshold = 0.3;

  // The minimum confidence score of face presence score in the face landmark
  // detection.
  float min_face_presence_confidence = 0.5;

  // The minimum confidence score of hand presence score in the hand landmark
  // detection.
  float min_hand_landmarks_confidence = 0.5;

  // The minimum confidence score for the pose detection to be considered
  // successful.
  float min_pose_detection_confidence = 0.5;

  // The minimum threshold for the pose suppression score in the pose detection.
  float min_pose_suppression_threshold = 0.3;

  // The minimum confidence score of pose presence score in the pose landmark
  // detection.
  float min_pose_presence_confidence = 0.5;

  // Whether to output face blendshapes classification. Face blendshapes are
  // used for rendering animations of the face.
  bool output_face_blendshapes = false;

  // Whether to output segmentation masks.
  bool output_pose_segmentation_masks = false;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  std::function<void(absl::StatusOr<HolisticLandmarkerResult>, const Image&,
                     int64_t)>
      result_callback = nullptr;
};

// Performs holistic landmarks detection on the given image.
//
// This API expects a pre-trained holistic landmarker model asset bundle.
//
// Inputs:
//   Image
//     - The image that holistic landmarks detection runs on.
//   std::optional<NormalizedRect>
//     - If provided, can be used to specify the rotation to apply to the image
//       before performing holistic landmarks detection, by setting its
//       'rotation' field in radians (e.g. 'M_PI / 2' for a 90° anti-clockwise
//       rotation). Note that specifying a region-of-interest using the
//       'x_center', 'y_center', 'width' and 'height' fields is NOT supported
//       and will result in an invalid argument error being returned.
// Outputs:
//   HolisticLandmarkerResult
//     - The holistic landmarks detection results.
class HolisticLandmarker : tasks::vision::core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates a HolisticLandmarker from a HolisticLandmarkerOptions to process
  // image data or streaming data. Holistic landmarker can be created with one
  // of the following three running modes:
  // 1) Image mode for detecting holistic landmarks on single image inputs.
  //    Users provide mediapipe::Image to the `Detect` method, and will receive
  //    the detected holistic landmarks results as the return value.
  // 2) Video mode for detecting holistic landmarks on the decoded frames of a
  //    video. Users call `DetectForVideo` method, and will receive the
  //    detected holistic landmarks results as the return value.
  // 3) Live stream mode for detecting holistic landmarks on the live stream of
  //    the input data, such as from camera. Users call `DetectAsync` to push
  //    the image data into the HolisticLandmarker. The detected results along
  //    with the input timestamp and the image that holistic landmarker runs on
  //    will be available in the result callback when the holistic landmarker
  //    finishes its processing.
  static absl::StatusOr<std::unique_ptr<HolisticLandmarker>> Create(
      std::unique_ptr<HolisticLandmarkerOptions> options);

  // Performs holistic landmarks detection on the given image.
  // Only use this method when the HolisticLandmarker is created with the image
  // running mode.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by setting
  // its 'rotation_degrees' field. Note that specifying a region-of-interest
  // using the 'region_of_interest' field is NOT supported and will result in
  // an invalid argument error being returned.
  //
  // The image can be of any size with format RGB or RGBA.
  absl::StatusOr<HolisticLandmarkerResult> Detect(
      Image image, const std::optional<core::ImageProcessingOptions>&
                       image_processing_options = std::nullopt);

  // Performs holistic landmarks detection on the provided video frame.
  // Only use this method when the HolisticLandmarker is created with the video
  // running mode.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by setting
  // its 'rotation_degrees' field. Note that specifying a region-of-interest
  // using the 'region_of_interest' field is NOT supported and will result in
  // an invalid argument error being returned.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide the video frame's timestamp (in milliseconds). The input
  // timestamps must be monotonically increasing.
  absl::StatusOr<HolisticLandmarkerResult> DetectForVideo(
      Image image, int64_t timestamp_ms,
      const std::optional<core::ImageProcessingOptions>&
          image_processing_options = std::nullopt);

  // Sends live image data to perform holistic landmarks detection, and the
  // results will be available via the "result_callback" provided in the
  // HolisticLandmarkerOptions. Only use this method when the HolisticLandmarker
  // is created with the live stream running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide a timestamp (in milliseconds) to indicate when the input image is
  // sent to the holistic landmarker. The input timestamps must be monotonically
  // increasing.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing detection, by setting
  // its 'rotation_degrees' field. Note that specifying a region-of-interest
  // using the 'region_of_interest' field is NOT supported and will result in
  // an invalid argument error being returned.
  //
  // The "result_callback" provides
  //   - A HolisticLandmarkerResult.
  //   - The const reference to the corresponding input image that the holistic
  //     landmarker runs on. Note that the const reference to the image will no
  //     longer be valid after the callback returns. To access the image data
  //     outside of the callback, callers need to make a copy of the image.
  //   - The input timestamp in milliseconds.
  absl::Status DetectAsync(Image image, int64_t timestamp_ms,
                           const std::optional<core::ImageProcessingOptions>&
                               image_processing_options = std::nullopt);

  // Shuts down the HolisticLandmarker when all works are done.
  absl::Status Close() { return runner_->Close(); }

 private:
  bool output_pose_segmentation_masks_ = false;
  bool output_face_blendshapes_ = false;
};

}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_HOLISTIC_LANDMARKER_HOLISTIC_LANDMARKER_H_
