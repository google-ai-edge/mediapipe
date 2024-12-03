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

#ifndef MEDIAPIPE_TASKS_CC_VISION_GESTURE_RECOGNIZRER_GESTURE_RECOGNIZER_H_
#define MEDIAPIPE_TASKS_CC_VISION_GESTURE_RECOGNIZRER_GESTURE_RECOGNIZER_H_

#include <cstdint>
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
#include "mediapipe/tasks/cc/vision/gesture_recognizer/gesture_recognizer_result.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace gesture_recognizer {

struct GestureRecognizerOptions {
  // Base options for configuring Task library, such as specifying the TfLite
  // model file with metadata, accelerator options, op resolver, etc.
  tasks::core::BaseOptions base_options;

  // The running mode of the task. Default to the image mode.
  // GestureRecognizer has three running modes:
  // 1) The image mode for recognizing hand gestures on single image inputs.
  // 2) The video mode for recognizing hand gestures on the decoded frames of a
  //    video.
  // 3) The live stream mode for recognizing hand gestures on the live stream of
  //    input data, such as from camera. In this mode, the "result_callback"
  //    below must be specified to receive the detection results asynchronously.
  core::RunningMode running_mode = core::RunningMode::IMAGE;

  // The maximum number of hands can be detected by the GestureRecognizer.
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

  // TODO  Note this option is subject to change.
  // Options for configuring the canned gestures classifier, such as score
  // threshold, allow list and deny list of gestures. The categories for canned
  // gesture classifiers are: ["None", "Closed_Fist", "Open_Palm",
  // "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
  components::processors::ClassifierOptions canned_gestures_classifier_options;

  // TODO  Note this option is subject to change.
  // Options for configuring the custom gestures classifier, such as score
  // threshold, allow list and deny list of gestures.
  components::processors::ClassifierOptions custom_gestures_classifier_options;

  // The user-defined result callback for processing live stream data.
  // The result callback should only be specified when the running mode is set
  // to RunningMode::LIVE_STREAM.
  std::function<void(absl::StatusOr<GestureRecognizerResult>, const Image&,
                     int64_t)>
      result_callback = nullptr;
};

// Performs hand gesture recognition on the given image.
//
// TODO  add the link to DevSite.
// This API expects a pre-trained hand gesture model asset bundle, or a custom
// one created using Model Maker. See <link to the DevSite documentation page>.
//
// Inputs:
//   Image
//     - The image that gesture recognition runs on.
//   std::optional<NormalizedRect>
//     - If provided, can be used to specify the rotation to apply to the image
//       before performing gesture recognition, by setting its 'rotation' field
//       in radians (e.g. 'M_PI / 2' for a 90° anti-clockwise rotation). Note
//       that specifying a region-of-interest using the 'x_center', 'y_center',
//       'width' and 'height' fields is NOT supported and will result in an
//       invalid argument error being returned.
// Outputs:
//   GestureRecognizerResult
//     - The hand gesture recognition results.
class GestureRecognizer : tasks::vision::core::BaseVisionTaskApi {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates a GestureRecognizer from a GestureRecognizerhOptions to process
  // image data or streaming data. Gesture recognizer can be created with one of
  // the following three running modes:
  // 1) Image mode for recognizing gestures on single image inputs.
  //    Users provide mediapipe::Image to the `Recognize` method, and will
  //    receive the recognized hand gesture results as the return value.
  // 2) Video mode for recognizing gestures on the decoded frames of a video.
  // 3) Live stream mode for recognizing gestures on the live stream of the
  //    input data, such as from camera. Users call `RecognizeAsync` to push the
  //    image data into the GestureRecognizer, the recognized results along with
  //    the input timestamp and the image that gesture recognizer runs on will
  //    be available in the result callback when the gesture recognizer finishes
  //    the work.
  static absl::StatusOr<std::unique_ptr<GestureRecognizer>> Create(
      std::unique_ptr<GestureRecognizerOptions> options);

  // Performs hand gesture recognition on the given image.
  // Only use this method when the GestureRecognizer is created with the image
  // running mode.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing recognition, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  //
  // The image can be of any size with format RGB or RGBA.
  // TODO: Describes how the input image will be preprocessed
  // after the yuv support is implemented.
  absl::StatusOr<GestureRecognizerResult> Recognize(
      Image image,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Performs gesture recognition on the provided video frame.
  // Only use this method when the GestureRecognizer is created with the video
  // running mode.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing recognition, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide the video frame's timestamp (in milliseconds). The input timestamps
  // must be monotonically increasing.
  absl::StatusOr<GestureRecognizerResult> RecognizeForVideo(
      Image image, int64_t timestamp_ms,
      std::optional<core::ImageProcessingOptions> image_processing_options =
          std::nullopt);

  // Sends live image data to perform gesture recognition, and the results will
  // be available via the "result_callback" provided in the
  // GestureRecognizerOptions. Only use this method when the GestureRecognizer
  // is created with the live stream running mode.
  //
  // The image can be of any size with format RGB or RGBA. It's required to
  // provide a timestamp (in milliseconds) to indicate when the input image is
  // sent to the gesture recognizer. The input timestamps must be monotonically
  // increasing.
  //
  // The optional 'image_processing_options' parameter can be used to specify
  // the rotation to apply to the image before performing recognition, by
  // setting its 'rotation_degrees' field. Note that specifying a
  // region-of-interest using the 'region_of_interest' field is NOT supported
  // and will result in an invalid argument error being returned.
  //
  // The "result_callback" provides
  //   - A vector of GestureRecognizerResult, each is the recognized results
  //     for a input frame.
  //   - The const reference to the corresponding input image that the gesture
  //     recognizer runs on. Note that the const reference to the image will no
  //     longer be valid when the callback returns. To access the image data
  //     outside of the callback, callers need to make a copy of the image.
  //   - The input timestamp in milliseconds.
  absl::Status RecognizeAsync(Image image, int64_t timestamp_ms,
                              std::optional<core::ImageProcessingOptions>
                                  image_processing_options = std::nullopt);

  // Shuts down the GestureRecognizer when all works are done.
  absl::Status Close() { return runner_->Close(); }
};

}  // namespace gesture_recognizer
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_GESTURE_RECOGNIZRER_GESTURE_RECOGNIZER_H_
