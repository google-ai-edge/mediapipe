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

#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"
#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker_result.h"
#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker_result_converter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_landmarker.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_landmarker_result.h"

using ::mediapipe::tasks::vision::holistic_landmarker::HolisticLandmarker;

struct MpHolisticLandmarkerInternal {
  std::unique_ptr<HolisticLandmarker> instance;
};

namespace mediapipe::tasks::c::vision::holistic_landmarker {

namespace {

using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::holistic_landmarker::HolisticLandmarker;
using CppHolisticLandmarkerResult =
    ::mediapipe::tasks::vision::holistic_landmarker::HolisticLandmarkerResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

HolisticLandmarker* GetCppLandmarker(MpHolisticLandmarkerPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "HolisticLandmarker is null.";
  return wrapper->instance.get();
}

}  // namespace

void CppConvertToHolisticLandmarkerOptions(
    const HolisticLandmarkerOptions& in,
    mediapipe::tasks::vision::holistic_landmarker::HolisticLandmarkerOptions*
        out) {
  out->min_face_detection_confidence = in.min_face_detection_confidence;
  out->min_face_suppression_threshold = in.min_face_suppression_threshold;
  out->min_face_presence_confidence = in.min_face_presence_confidence;
  out->min_hand_landmarks_confidence = in.min_hand_landmarks_confidence;
  out->min_pose_detection_confidence = in.min_pose_detection_confidence;
  out->min_pose_suppression_threshold = in.min_pose_suppression_threshold;
  out->min_pose_presence_confidence = in.min_pose_presence_confidence;
  out->output_face_blendshapes = in.output_face_blendshapes;
  out->output_pose_segmentation_masks = in.output_pose_segmentation_masks;
}

absl::Status CppHolisticLandmarkerCreate(
    const HolisticLandmarkerOptions& options,
    MpHolisticLandmarkerPtr* landmarker) {
  auto cpp_options =
      std::make_unique<::mediapipe::tasks::vision::holistic_landmarker::
                           HolisticLandmarkerOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToHolisticLandmarkerOptions(options, cpp_options.get());
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  // Enable callback for processing live stream data when the running mode is
  // set to RunningMode::LIVE_STREAM.
  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      return absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
    }

    HolisticLandmarkerOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](
            absl::StatusOr<CppHolisticLandmarkerResult> cpp_result,
            const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }

          HolisticLandmarkerResult result;
          CppConvertToHolisticLandmarkerResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppCloseHolisticLandmarkerResult(&result);
        };
  }

  auto cpp_landmarker = HolisticLandmarker::Create(std::move(cpp_options));
  if (!cpp_landmarker.ok()) {
    return cpp_landmarker.status();
  }
  *landmarker =
      new MpHolisticLandmarkerInternal{.instance = std::move(*cpp_landmarker)};
  return absl::OkStatus();
}

absl::Status CppHolisticLandmarkerDetect(
    MpHolisticLandmarkerPtr landmarker, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    HolisticLandmarkerResult* result) {
  auto* cpp_landmarker = GetCppLandmarker(landmarker);
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    core::CppConvertToImageProcessingOptions(*image_processing_options,
                                             &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result =
      cpp_landmarker->Detect(ToImage(image), cpp_image_processing_options);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToHolisticLandmarkerResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppHolisticLandmarkerDetectForVideo(
    MpHolisticLandmarkerPtr landmarker, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, HolisticLandmarkerResult* result) {
  auto* cpp_landmarker = GetCppLandmarker(landmarker);
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    core::CppConvertToImageProcessingOptions(*image_processing_options,
                                             &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result = cpp_landmarker->DetectForVideo(
      ToImage(image), timestamp_ms, cpp_image_processing_options);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToHolisticLandmarkerResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppHolisticLandmarkerDetectAsync(
    MpHolisticLandmarkerPtr landmarker, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  auto* cpp_landmarker = GetCppLandmarker(landmarker);
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    core::CppConvertToImageProcessingOptions(*image_processing_options,
                                             &options);
    cpp_image_processing_options = options;
  }
  return cpp_landmarker->DetectAsync(ToImage(image), timestamp_ms,
                                     cpp_image_processing_options);
}

absl::Status CppHolisticLandmarkerClose(MpHolisticLandmarkerPtr landmarker) {
  auto* cpp_landmarker = GetCppLandmarker(landmarker);
  auto result = cpp_landmarker->Close();
  if (!result.ok()) {
    return result;
  }
  delete landmarker;
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::vision::holistic_landmarker

extern "C" {

MpStatus MpHolisticLandmarkerCreate(struct HolisticLandmarkerOptions* options,
                                    MpHolisticLandmarkerPtr* landmarker,
                                    char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::holistic_landmarker::
      CppHolisticLandmarkerCreate(*options, landmarker);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MpStatus MpHolisticLandmarkerDetectImage(
    MpHolisticLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    HolisticLandmarkerResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::holistic_landmarker::
      CppHolisticLandmarkerDetect(landmarker, image, image_processing_options,
                                  result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MpStatus MpHolisticLandmarkerDetectForVideo(
    MpHolisticLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, HolisticLandmarkerResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::holistic_landmarker::
      CppHolisticLandmarkerDetectForVideo(
          landmarker, image, image_processing_options, timestamp_ms, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MpStatus MpHolisticLandmarkerDetectAsync(
    MpHolisticLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::holistic_landmarker::
      CppHolisticLandmarkerDetectAsync(landmarker, image,
                                       image_processing_options, timestamp_ms);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

void MpHolisticLandmarkerCloseResult(HolisticLandmarkerResult* result) {
  mediapipe::tasks::c::vision::holistic_landmarker::
      CppCloseHolisticLandmarkerResult(result);
}

MpStatus MpHolisticLandmarkerClose(MpHolisticLandmarkerPtr landmarker,
                                   char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::holistic_landmarker::
      CppHolisticLandmarkerClose(landmarker);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
