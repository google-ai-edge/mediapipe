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

#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"

#include <cstdint>
#include <cstdlib>
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
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker_result.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker_result_converter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

using ::mediapipe::tasks::vision::face_landmarker::FaceLandmarker;

struct MpFaceLandmarkerInternal {
  std::unique_ptr<FaceLandmarker> instance;
};

namespace mediapipe::tasks::c::vision::face_landmarker {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseFaceLandmarkerResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToFaceLandmarkerResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::CreateImageFromBuffer;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::face_landmarker::FaceLandmarker;
using CppFaceLandmarkerResult =
    ::mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

FaceLandmarker* GetCppLandmarker(MpFaceLandmarkerPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "FaceLandmarker is null.";
  return wrapper->instance.get();
}

}  // namespace

void CppConvertToFaceLandmarkerOptions(
    const FaceLandmarkerOptions& in,
    mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions* out) {
  out->num_faces = in.num_faces;
  out->min_face_detection_confidence = in.min_face_detection_confidence;
  out->min_face_presence_confidence = in.min_face_presence_confidence;
  out->min_tracking_confidence = in.min_tracking_confidence;
  out->output_face_blendshapes = in.output_face_blendshapes;
  out->output_facial_transformation_matrixes =
      in.output_facial_transformation_matrixes;
}

absl::Status CppFaceLandmarkerCreate(const FaceLandmarkerOptions& options,
                                     MpFaceLandmarkerPtr* landmarker) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToFaceLandmarkerOptions(options, cpp_options.get());
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  // Enable callback for processing live stream data when the running mode is
  // set to RunningMode::LIVE_STREAM.
  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      return absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
    }

    FaceLandmarkerOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppFaceLandmarkerResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }

          FaceLandmarkerResult result;
          CppConvertToFaceLandmarkerResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppCloseFaceLandmarkerResult(&result);
        };
  }

  auto cpp_landmarker = FaceLandmarker::Create(std::move(cpp_options));
  if (!cpp_landmarker.ok()) {
    return cpp_landmarker.status();
  }
  *landmarker =
      new MpFaceLandmarkerInternal{.instance = std::move(*cpp_landmarker)};
  return absl::OkStatus();
}

absl::Status CppFaceLandmarkerDetect(
    MpFaceLandmarkerPtr landmarker, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    FaceLandmarkerResult* result) {
  auto* cpp_landmarker = GetCppLandmarker(landmarker);
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
  CppConvertToFaceLandmarkerResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppFaceLandmarkerDetectForVideo(
    MpFaceLandmarkerPtr landmarker, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, FaceLandmarkerResult* result) {
  auto* cpp_landmarker = GetCppLandmarker(landmarker);
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
  CppConvertToFaceLandmarkerResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppFaceLandmarkerDetectAsync(
    MpFaceLandmarkerPtr landmarker, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  auto* cpp_landmarker = GetCppLandmarker(landmarker);
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  return cpp_landmarker->DetectAsync(ToImage(image), timestamp_ms,
                                     cpp_image_processing_options);
}

void CppFaceLandmarkerCloseResult(FaceLandmarkerResult* result) {
  CppCloseFaceLandmarkerResult(result);
}

absl::Status CppFaceLandmarkerClose(MpFaceLandmarkerPtr landmarker) {
  auto* cpp_landmarker = GetCppLandmarker(landmarker);
  auto result = cpp_landmarker->Close();
  if (!result.ok()) {
    return result;
  }
  delete landmarker;
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::vision::face_landmarker

extern "C" {

MpStatus MpFaceLandmarkerCreate(struct FaceLandmarkerOptions* options,
                                MpFaceLandmarkerPtr* landmarker,
                                char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::vision::face_landmarker::CppFaceLandmarkerCreate(
          *options, landmarker);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MpStatus MpFaceLandmarkerDetectImage(
    MpFaceLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    FaceLandmarkerResult* result, char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::vision::face_landmarker::CppFaceLandmarkerDetect(
          landmarker, image, image_processing_options, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MpStatus MpFaceLandmarkerDetectForVideo(
    MpFaceLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, FaceLandmarkerResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::face_landmarker::
      CppFaceLandmarkerDetectForVideo(
          landmarker, image, image_processing_options, timestamp_ms, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MpStatus MpFaceLandmarkerDetectAsync(
    MpFaceLandmarkerPtr landmarker, MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::face_landmarker::
      CppFaceLandmarkerDetectAsync(landmarker, image, image_processing_options,
                                   timestamp_ms);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

void MpFaceLandmarkerCloseResult(FaceLandmarkerResult* result) {
  mediapipe::tasks::c::vision::face_landmarker::CppFaceLandmarkerCloseResult(
      result);
}

MpStatus MpFaceLandmarkerClose(MpFaceLandmarkerPtr landmarker,
                               char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::vision::face_landmarker::CppFaceLandmarkerClose(
          landmarker);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
