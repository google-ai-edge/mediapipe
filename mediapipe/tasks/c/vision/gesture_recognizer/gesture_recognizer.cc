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
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/c/components/processors/classifier_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer_result.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer_result_converter.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/gesture_recognizer.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/gesture_recognizer_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe::tasks::c::vision::gesture_recognizer {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseGestureRecognizerResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToGestureRecognizerResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToClassifierOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::vision::CreateImageFromBuffer;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::gesture_recognizer::GestureRecognizer;
typedef ::mediapipe::tasks::vision::gesture_recognizer::GestureRecognizerResult
    CppGestureRecognizerResult;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
}

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

GestureRecognizer* CppGestureRecognizerCreate(
    const GestureRecognizerOptions& options, char** error_msg) {
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
      CppProcessError(status, error_msg);
      return nullptr;
    }

    GestureRecognizerOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppGestureRecognizerResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          char* error_msg = nullptr;

          if (!cpp_result.ok()) {
            ABSL_LOG(ERROR) << "Recognition failed: " << cpp_result.status();
            CppProcessError(cpp_result.status(), &error_msg);
            result_callback(nullptr, nullptr, timestamp, error_msg);
            free(error_msg);
            return;
          }

          // Result is valid for the lifetime of the callback function.
          auto result = std::make_unique<GestureRecognizerResult>();
          CppConvertToGestureRecognizerResult(*cpp_result, result.get());

          const auto& image_frame = image.GetImageFrameSharedPtr();
          const MpImage mp_image = {
              .type = MpImage::IMAGE_FRAME,
              .image_frame = {
                  .format = static_cast<::ImageFormat>(image_frame->Format()),
                  .image_buffer = image_frame->PixelData(),
                  .width = image_frame->Width(),
                  .height = image_frame->Height()}};

          result_callback(result.release(), &mp_image, timestamp,
                          /* error_msg= */ nullptr);
        };
  }

  auto recognizer = GestureRecognizer::Create(std::move(cpp_options));
  if (!recognizer.ok()) {
    ABSL_LOG(ERROR) << "Failed to create GestureRecognizer: "
                    << recognizer.status();
    CppProcessError(recognizer.status(), error_msg);
    return nullptr;
  }
  return recognizer->release();
}

int CppGestureRecognizerRecognize(void* recognizer, const MpImage* image,
                                  GestureRecognizerResult* result,
                                  char** error_msg) {
  if (image->type == MpImage::GPU_BUFFER) {
    const absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet.");

    ABSL_LOG(ERROR) << "Recognition failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image->image_frame.format),
      image->image_frame.image_buffer, image->image_frame.width,
      image->image_frame.height);

  if (!img.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_recognizer = static_cast<GestureRecognizer*>(recognizer);
  auto cpp_result = cpp_recognizer->Recognize(*img);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Recognition failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToGestureRecognizerResult(*cpp_result, result);
  return 0;
}

int CppGestureRecognizerRecognizeForVideo(void* recognizer,
                                          const MpImage* image,
                                          int64_t timestamp_ms,
                                          GestureRecognizerResult* result,
                                          char** error_msg) {
  if (image->type == MpImage::GPU_BUFFER) {
    absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet");

    ABSL_LOG(ERROR) << "Recognition failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image->image_frame.format),
      image->image_frame.image_buffer, image->image_frame.width,
      image->image_frame.height);

  if (!img.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_recognizer = static_cast<GestureRecognizer*>(recognizer);
  auto cpp_result = cpp_recognizer->RecognizeForVideo(*img, timestamp_ms);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Recognition failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToGestureRecognizerResult(*cpp_result, result);
  return 0;
}

int CppGestureRecognizerRecognizeAsync(void* recognizer, const MpImage* image,
                                       int64_t timestamp_ms, char** error_msg) {
  if (image->type == MpImage::GPU_BUFFER) {
    absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet");

    ABSL_LOG(ERROR) << "Recognition failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image->image_frame.format),
      image->image_frame.image_buffer, image->image_frame.width,
      image->image_frame.height);

  if (!img.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_recognizer = static_cast<GestureRecognizer*>(recognizer);
  auto cpp_result = cpp_recognizer->RecognizeAsync(*img, timestamp_ms);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Data preparation for the gesture recognition failed: "
                    << cpp_result;
    return CppProcessError(cpp_result, error_msg);
  }
  return 0;
}

void CppGestureRecognizerCloseResult(GestureRecognizerResult* result) {
  CppCloseGestureRecognizerResult(result);
}

int CppGestureRecognizerClose(void* recognizer, char** error_msg) {
  auto cpp_recognizer = static_cast<GestureRecognizer*>(recognizer);
  auto result = cpp_recognizer->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close GestureRecognizer: " << result;
    return CppProcessError(result, error_msg);
  }
  delete cpp_recognizer;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::gesture_recognizer

extern "C" {

void* gesture_recognizer_create(struct GestureRecognizerOptions* options,
                                char** error_msg) {
  return mediapipe::tasks::c::vision::gesture_recognizer::
      CppGestureRecognizerCreate(*options, error_msg);
}

int gesture_recognizer_recognize_image(void* recognizer, const MpImage* image,
                                       GestureRecognizerResult* result,
                                       char** error_msg) {
  return mediapipe::tasks::c::vision::gesture_recognizer::
      CppGestureRecognizerRecognize(recognizer, image, result, error_msg);
}

int gesture_recognizer_recognize_for_video(void* recognizer,
                                           const MpImage* image,
                                           int64_t timestamp_ms,
                                           GestureRecognizerResult* result,
                                           char** error_msg) {
  return mediapipe::tasks::c::vision::gesture_recognizer::
      CppGestureRecognizerRecognizeForVideo(recognizer, image, timestamp_ms,
                                            result, error_msg);
}

int gesture_recognizer_recognize_async(void* recognizer, const MpImage* image,
                                       int64_t timestamp_ms, char** error_msg) {
  return mediapipe::tasks::c::vision::gesture_recognizer::
      CppGestureRecognizerRecognizeAsync(recognizer, image, timestamp_ms,
                                         error_msg);
}

void gesture_recognizer_close_result(GestureRecognizerResult* result) {
  mediapipe::tasks::c::vision::gesture_recognizer::
      CppGestureRecognizerCloseResult(result);
}

int gesture_recognizer_close(void* recognizer, char** error_ms) {
  return mediapipe::tasks::c::vision::gesture_recognizer::
      CppGestureRecognizerClose(recognizer, error_ms);
}

}  // extern "C"
