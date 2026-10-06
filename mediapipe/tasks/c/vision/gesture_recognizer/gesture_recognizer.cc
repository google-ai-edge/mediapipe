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

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/components/processors/classifier_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
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

using ::mediapipe::tasks::vision::gesture_recognizer::GestureRecognizer;

struct MpGestureRecognizerInternal {
  std::unique_ptr<GestureRecognizer> instance;
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
using CppMpGestureRecognizerResult =
    ::mediapipe::tasks::vision::gesture_recognizer::GestureRecognizerResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

GestureRecognizer* GetCppRecognizer(MpGestureRecognizerPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "GestureRecognizer is null.";
  return wrapper->instance.get();
}

}  // namespace

void CppConvertToGestureRecognizerOptions(
    const MpGestureRecognizerOptions& in,
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

absl::Status CppMpGestureRecognizerCreate(
    const MpGestureRecognizerOptions& options,
    MpGestureRecognizerPtr* recognizer) {
  auto cpp_options =
      std::make_unique<::mediapipe::tasks::vision::gesture_recognizer::
                           GestureRecognizerOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToGestureRecognizerOptions(options, cpp_options.get());
  cpp_options->running_mode =
      static_cast<::mediapipe::tasks::vision::core::RunningMode>(
          options.running_mode);

  // Enable callback for processing live stream data when the running mode is
  // set to MpRunningMode::MP_RUNNING_MODE_LIVE_STREAM.
  if (cpp_options->running_mode == MpRunningMode::MP_RUNNING_MODE_LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      return absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
    }

    MpGestureRecognizerOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](
            absl::StatusOr<CppMpGestureRecognizerResult> cpp_result,
            const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }

          MpGestureRecognizerResult result;
          CppConvertToGestureRecognizerResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppCloseGestureRecognizerResult(&result);
        };
  }

  auto cpp_recognizer = GestureRecognizer::Create(std::move(cpp_options));
  if (!cpp_recognizer.ok()) {
    return cpp_recognizer.status();
  }
  *recognizer =
      new MpGestureRecognizerInternal{.instance = std::move(*cpp_recognizer)};
  return absl::OkStatus();
}

absl::Status CppMpGestureRecognizerRecognizeImage(
    MpGestureRecognizerPtr recognizer, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    MpGestureRecognizerResult* result) {
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  GestureRecognizer* cpp_recognizer = GetCppRecognizer(recognizer);
  auto cpp_result =
      cpp_recognizer->Recognize(ToImage(image), cpp_image_processing_options);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToGestureRecognizerResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppMpGestureRecognizerRecognizeForVideo(
    MpGestureRecognizerPtr recognizer, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, MpGestureRecognizerResult* result) {
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  GestureRecognizer* cpp_recognizer = GetCppRecognizer(recognizer);
  auto cpp_result = cpp_recognizer->RecognizeForVideo(
      ToImage(image), timestamp_ms, cpp_image_processing_options);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToGestureRecognizerResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppMpGestureRecognizerRecognizeAsync(
    MpGestureRecognizerPtr recognizer, const MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  GestureRecognizer* cpp_recognizer = GetCppRecognizer(recognizer);
  return cpp_recognizer->RecognizeAsync(ToImage(image), timestamp_ms,
                                        cpp_image_processing_options);
}

void CppMpGestureRecognizerCloseResult(MpGestureRecognizerResult* result) {
  CppCloseGestureRecognizerResult(result);
}

absl::Status CppMpGestureRecognizerClose(MpGestureRecognizerPtr recognizer) {
  auto cpp_recognizer = GetCppRecognizer(recognizer);
  auto result = cpp_recognizer->Close();
  if (!result.ok()) {
    return result;
  }
  delete recognizer;
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::vision::gesture_recognizer

extern "C" {

MP_EXPORT MpStatus MpGestureRecognizerCreate(
    const struct MpGestureRecognizerOptions* options,
    MpGestureRecognizerPtr* recognizer, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::gesture_recognizer::
      CppMpGestureRecognizerCreate(*options, recognizer);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpGestureRecognizerRecognizeImage(
    MpGestureRecognizerPtr recognizer, MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    MpGestureRecognizerResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::gesture_recognizer::
      CppMpGestureRecognizerRecognizeImage(recognizer, image,
                                           image_processing_options, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpGestureRecognizerRecognizeForVideo(
    MpGestureRecognizerPtr recognizer, MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, MpGestureRecognizerResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::gesture_recognizer::
      CppMpGestureRecognizerRecognizeForVideo(
          recognizer, image, image_processing_options, timestamp_ms, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpGestureRecognizerRecognizeAsync(
    MpGestureRecognizerPtr recognizer, MpImagePtr image,
    const MpImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::gesture_recognizer::
      CppMpGestureRecognizerRecognizeAsync(
          recognizer, image, image_processing_options, timestamp_ms);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpGestureRecognizerCloseResult(
    MpGestureRecognizerResult* result) {
  mediapipe::tasks::c::vision::gesture_recognizer::
      CppMpGestureRecognizerCloseResult(result);
}

MP_EXPORT MpStatus MpGestureRecognizerClose(MpGestureRecognizerPtr recognizer,
                                            char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::gesture_recognizer::
      CppMpGestureRecognizerClose(recognizer);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
