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

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
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
      landmarker;
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

}  // namespace

void CppConvertToHandLandmarkerOptions(
    const HandLandmarkerOptions& in,
    mediapipe::tasks::vision::hand_landmarker::HandLandmarkerOptions* out) {
  out->num_hands = in.num_hands;
  out->min_hand_detection_confidence = in.min_hand_detection_confidence;
  out->min_hand_presence_confidence = in.min_hand_presence_confidence;
  out->min_tracking_confidence = in.min_tracking_confidence;
}

MpStatus CppHandLandmarkerCreate(const HandLandmarkerOptions& options,
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
      const absl::Status status = absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
      ABSL_LOG(ERROR) << "Failed to create HandLandmarker: " << status;
      return ToMpStatus(status);
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
    ABSL_LOG(ERROR) << "Failed to create HandLandmarker: "
                    << cpp_landmarker.status();
    return ToMpStatus(cpp_landmarker.status());
  }
  *landmarker =
      new MpHandLandmarkerInternal{.landmarker = std::move(*cpp_landmarker)};
  return kMpOk;
}

MpStatus CppHandLandmarkerDetect(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    HandLandmarkerResult* result) {
  auto cpp_landmarker = landmarker->landmarker.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result =
      cpp_landmarker->Detect(ToImage(image), cpp_image_processing_options);

  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Recognition failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToHandLandmarkerResult(*cpp_result, result);
  return kMpOk;
}

MpStatus CppHandLandmarkerDetectForVideo(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, HandLandmarkerResult* result) {
  auto cpp_landmarker = landmarker->landmarker.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result = cpp_landmarker->DetectForVideo(
      ToImage(image), timestamp_ms, cpp_image_processing_options);

  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Recognition failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToHandLandmarkerResult(*cpp_result, result);
  return kMpOk;
}

MpStatus CppHandLandmarkerDetectAsync(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  auto cpp_landmarker = landmarker->landmarker.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result = cpp_landmarker->DetectAsync(ToImage(image), timestamp_ms,
                                                cpp_image_processing_options);

  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Data preparation for the landmark detection failed: "
                    << cpp_result;
    return ToMpStatus(cpp_result);
  }
  return kMpOk;
}

void CppHandLandmarkerCloseResult(HandLandmarkerResult* result) {
  CppCloseHandLandmarkerResult(result);
}

MpStatus CppHandLandmarkerClose(MpHandLandmarkerPtr landmarker) {
  auto cpp_landmarker = landmarker->landmarker.get();
  auto result = cpp_landmarker->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close HandLandmarker: " << result;
    return ToMpStatus(result);
  }
  delete landmarker;
  return kMpOk;
}

}  // namespace mediapipe::tasks::c::vision::hand_landmarker

extern "C" {

MP_EXPORT MpStatus MpHandLandmarkerCreate(struct HandLandmarkerOptions* options,
                                          MpHandLandmarkerPtr* landmarker) {
  return mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerCreate(
      *options, landmarker);
}

MP_EXPORT MpStatus MpHandLandmarkerDetectImage(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    HandLandmarkerResult* result) {
  return mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerDetect(
      landmarker, image, image_processing_options, result);
}

MP_EXPORT MpStatus MpHandLandmarkerDetectForVideo(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, HandLandmarkerResult* result) {
  return mediapipe::tasks::c::vision::hand_landmarker::
      CppHandLandmarkerDetectForVideo(
          landmarker, image, image_processing_options, timestamp_ms, result);
}

MP_EXPORT MpStatus MpHandLandmarkerDetectAsync(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  return mediapipe::tasks::c::vision::hand_landmarker::
      CppHandLandmarkerDetectAsync(landmarker, image, image_processing_options,
                                   timestamp_ms);
}

MP_EXPORT void MpHandLandmarkerCloseResult(HandLandmarkerResult* result) {
  mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerCloseResult(
      result);
}

MP_EXPORT MpStatus MpHandLandmarkerClose(MpHandLandmarkerPtr landmarker) {
  return mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerClose(
      landmarker);
}

}  // extern "C"
