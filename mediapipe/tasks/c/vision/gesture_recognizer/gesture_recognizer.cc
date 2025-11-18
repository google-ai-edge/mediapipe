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

#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/components/processors/classifier_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer_result.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer_result_converter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/gesture_recognizer.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/gesture_recognizer_result.h"

struct MpGestureRecognizerInternal {
  std::unique_ptr<
      ::mediapipe::tasks::vision::gesture_recognizer::GestureRecognizer>
      recognizer;
};

namespace mediapipe::tasks::c::vision::gesture_recognizer {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseGestureRecognizerResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToGestureRecognizerResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToClassifierOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::gesture_recognizer::GestureRecognizer;
using CppGestureRecognizerResult =
    ::mediapipe::tasks::vision::gesture_recognizer::GestureRecognizerResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

}  // namespace

void CppConvertToGestureRecognizerOptions(
    const GestureRecognizerOptions& in,
    mediapipe::tasks::vision::gesture_recognizer::GestureRecognizerOptions*
        out) {
  out->num_hands = in.num_hands;
  out->min_hand_detection_confidence = in.min_hand_detection_confidence;
  out->min_hand_presence_confidence = in.min_hand_presence_confidence;
  out->min_tracking_confidence = in.min_tracking_confidence;
  CppConvertToClassifierOptions(in.canned_gestures_classifier_options,
                                &out->canned_gestures_classifier_options);
  CppConvertToClassifierOptions(in.custom_gestures_classifier_options,
                                &out->custom_gestures_classifier_options);
}

MpStatus CppMpGestureRecognizerCreate(const GestureRecognizerOptions& options,
                                      MpGestureRecognizerPtr* recognizer) {
  auto cpp_options =
      std::make_unique<::mediapipe::tasks::vision::gesture_recognizer::
                           GestureRecognizerOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToGestureRecognizerOptions(options, cpp_options.get());
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  // Enable callback for processing live stream data when the running mode is
  // set to RunningMode::LIVE_STREAM.
  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      const absl::Status status = absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
      ABSL_LOG(ERROR) << "Failed to create GestureRecognizer: " << status;
      return ToMpStatus(status);
    }

    GestureRecognizerOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppGestureRecognizerResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            ABSL_LOG(ERROR) << "Recognition failed: " << cpp_result.status();
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }

          GestureRecognizerResult result;
          CppConvertToGestureRecognizerResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppCloseGestureRecognizerResult(&result);
        };
  }

  auto cpp_recognizer = GestureRecognizer::Create(std::move(cpp_options));
  if (!cpp_recognizer.ok()) {
    ABSL_LOG(ERROR) << "Failed to create GestureRecognizer: "
                    << cpp_recognizer.status();
    return ToMpStatus(cpp_recognizer.status());
  }
  *recognizer =
      new MpGestureRecognizerInternal{.recognizer = std::move(*cpp_recognizer)};
  return kMpOk;
}

MpStatus CppMpGestureRecognizerRecognizeImage(
    MpGestureRecognizerPtr recognizer, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    GestureRecognizerResult* result) {
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  GestureRecognizer* cpp_recognizer = recognizer->recognizer.get();
  auto cpp_result =
      cpp_recognizer->Recognize(ToImage(image), cpp_image_processing_options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Recognition failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToGestureRecognizerResult(*cpp_result, result);
  return kMpOk;
}

MpStatus CppMpGestureRecognizerRecognizeForVideo(
    MpGestureRecognizerPtr recognizer, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, GestureRecognizerResult* result) {
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  GestureRecognizer* cpp_recognizer = recognizer->recognizer.get();
  auto cpp_result = cpp_recognizer->RecognizeForVideo(
      ToImage(image), timestamp_ms, cpp_image_processing_options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Recognition failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToGestureRecognizerResult(*cpp_result, result);
  return kMpOk;
}

MpStatus CppMpGestureRecognizerRecognizeAsync(
    MpGestureRecognizerPtr recognizer, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  GestureRecognizer* cpp_recognizer = recognizer->recognizer.get();
  auto cpp_result = cpp_recognizer->RecognizeAsync(
      ToImage(image), timestamp_ms, cpp_image_processing_options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Data preparation for the gesture recognition failed: "
                    << cpp_result;
    return ToMpStatus(cpp_result);
  }
  return kMpOk;
}

void CppMpGestureRecognizerCloseResult(GestureRecognizerResult* result) {
  CppCloseGestureRecognizerResult(result);
}

MpStatus CppMpGestureRecognizerClose(MpGestureRecognizerPtr recognizer) {
  auto cpp_recognizer = recognizer->recognizer.get();
  auto result = cpp_recognizer->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close GestureRecognizer: " << result;
    return ToMpStatus(result);
  }
  delete recognizer;
  return kMpOk;
}

}  // namespace mediapipe::tasks::c::vision::gesture_recognizer

extern "C" {

MP_EXPORT MpStatus
MpGestureRecognizerCreate(const struct GestureRecognizerOptions* options,
                          MpGestureRecognizerPtr* recognizer) {
  return mediapipe::tasks::c::vision::gesture_recognizer::
      CppMpGestureRecognizerCreate(*options, recognizer);
}

MP_EXPORT MpStatus MpGestureRecognizerRecognizeImage(
    MpGestureRecognizerPtr recognizer, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    GestureRecognizerResult* result) {
  return mediapipe::tasks::c::vision::gesture_recognizer::
      CppMpGestureRecognizerRecognizeImage(recognizer, image,
                                           image_processing_options, result);
}

MP_EXPORT MpStatus MpGestureRecognizerRecognizeForVideo(
    MpGestureRecognizerPtr recognizer, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, GestureRecognizerResult* result) {
  return mediapipe::tasks::c::vision::gesture_recognizer::
      CppMpGestureRecognizerRecognizeForVideo(
          recognizer, image, image_processing_options, timestamp_ms, result);
}

MP_EXPORT MpStatus MpGestureRecognizerRecognizeAsync(
    MpGestureRecognizerPtr recognizer, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  return mediapipe::tasks::c::vision::gesture_recognizer::
      CppMpGestureRecognizerRecognizeAsync(
          recognizer, image, image_processing_options, timestamp_ms);
}

MP_EXPORT void MpGestureRecognizerCloseResult(GestureRecognizerResult* result) {
  mediapipe::tasks::c::vision::gesture_recognizer::
      CppMpGestureRecognizerCloseResult(result);
}

MP_EXPORT MpStatus MpGestureRecognizerClose(MpGestureRecognizerPtr recognizer) {
  return mediapipe::tasks::c::vision::gesture_recognizer::
      CppMpGestureRecognizerClose(recognizer);
}

}  // extern "C"
