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

#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result_converter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

struct MpHandLandmarkerInternal {
  std::unique_ptr<::mediapipe::tasks::vision::hand_landmarker::HandLandmarker>
      instance;
};

namespace mediapipe::tasks::c::vision::hand_landmarker {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseHandLandmarkerResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToHandLandmarkerResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::hand_landmarker::HandLandmarker;
using CppHandLandmarkerResult =
    ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

HandLandmarker* GetCppLandmarker(MpHandLandmarkerPtr landmarker) {
  ABSL_CHECK(landmarker != nullptr) << "HandLandmarker is null.";
  return landmarker->instance.get();
}

}  // namespace

void CppConvertToHandLandmarkerOptions(
    const HandLandmarkerOptions& in,
    mediapipe::tasks::vision::hand_landmarker::HandLandmarkerOptions* out) {
  out->num_hands = in.num_hands;
  out->min_hand_detection_confidence = in.min_hand_detection_confidence;
  out->min_hand_presence_confidence = in.min_hand_presence_confidence;
  out->min_tracking_confidence = in.min_tracking_confidence;
}

absl::Status CppHandLandmarkerCreate(const HandLandmarkerOptions& options,
                                     MpHandLandmarkerPtr* landmarker) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToHandLandmarkerOptions(options, cpp_options.get());
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  // Enable callback for processing live stream data when the running mode is
  // set to RunningMode::LIVE_STREAM.
  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      return absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
    }

    HandLandmarkerOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppHandLandmarkerResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }

          HandLandmarkerResult result;
          CppConvertToHandLandmarkerResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppCloseHandLandmarkerResult(&result);
        };
  }

  auto cpp_landmarker = HandLandmarker::Create(std::move(cpp_options));
  if (!cpp_landmarker.ok()) {
    return cpp_landmarker.status();
  }
  *landmarker =
      new MpHandLandmarkerInternal{.instance = std::move(*cpp_landmarker)};
  return absl::OkStatus();
}

absl::Status CppHandLandmarkerDetect(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    HandLandmarkerResult* result) {
  auto cpp_landmarker = GetCppLandmarker(landmarker);
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result =
      cpp_landmarker->Detect(ToImage(image), cpp_image_processing_options);

  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToHandLandmarkerResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppHandLandmarkerDetectForVideo(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, HandLandmarkerResult* result) {
  auto cpp_landmarker = GetCppLandmarker(landmarker);
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result = cpp_landmarker->DetectForVideo(
      ToImage(image), timestamp_ms, cpp_image_processing_options);

  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToHandLandmarkerResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppHandLandmarkerDetectAsync(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  auto cpp_landmarker = GetCppLandmarker(landmarker);
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result = cpp_landmarker->DetectAsync(ToImage(image), timestamp_ms,
                                                cpp_image_processing_options);

  return cpp_result;
}

void CppHandLandmarkerCloseResult(HandLandmarkerResult* result) {
  CppCloseHandLandmarkerResult(result);
}

absl::Status CppHandLandmarkerClose(MpHandLandmarkerPtr landmarker) {
  auto cpp_landmarker = GetCppLandmarker(landmarker);
  auto result = cpp_landmarker->Close();
  if (!result.ok()) {
    return result;
  }
  delete landmarker;
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::vision::hand_landmarker

extern "C" {

MP_EXPORT MpStatus MpHandLandmarkerCreate(struct HandLandmarkerOptions* options,
                                          MpHandLandmarkerPtr* landmarker,
                                          char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerCreate(
          *options, landmarker);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpHandLandmarkerDetectImage(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    HandLandmarkerResult* result, char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerDetect(
          landmarker, image, image_processing_options, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpHandLandmarkerDetectForVideo(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, HandLandmarkerResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::hand_landmarker::
      CppHandLandmarkerDetectForVideo(
          landmarker, image, image_processing_options, timestamp_ms, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpHandLandmarkerDetectAsync(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::hand_landmarker::
      CppHandLandmarkerDetectAsync(landmarker, image, image_processing_options,
                                   timestamp_ms);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpHandLandmarkerCloseResult(HandLandmarkerResult* result) {
  mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerCloseResult(
      result);
}

MP_EXPORT MpStatus MpHandLandmarkerClose(MpHandLandmarkerPtr landmarker,
                                         char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerClose(
          landmarker);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
