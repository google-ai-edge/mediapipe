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

#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker.h"

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
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker_result.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker_result_converter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_result.h"

struct MpPoseLandmarkerInternal {
  std::unique_ptr<::mediapipe::tasks::vision::pose_landmarker::PoseLandmarker>
      landmarker;
};

namespace mediapipe::tasks::c::vision::pose_landmarker {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppClosePoseLandmarkerResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToPoseLandmarkerResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::pose_landmarker::PoseLandmarker;
using CppPoseLandmarkerResult =
    ::mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;

}  // namespace

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

void CppConvertToPoseLandmarkerOptions(
    const PoseLandmarkerOptions& in,
    mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerOptions* out) {
  out->num_poses = in.num_poses;
  out->min_pose_detection_confidence = in.min_pose_detection_confidence;
  out->min_pose_presence_confidence = in.min_pose_presence_confidence;
  out->min_tracking_confidence = in.min_tracking_confidence;
  out->output_segmentation_masks = in.output_segmentation_masks;
}

MpStatus CppPoseLandmarkerCreate(const PoseLandmarkerOptions& options,
                                 MpPoseLandmarkerPtr* landmarker_out) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToPoseLandmarkerOptions(options, cpp_options.get());
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  // Enable callback for processing live stream data when the running mode is
  // set to RunningMode::LIVE_STREAM.
  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      const absl::Status status = absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
      ABSL_LOG(ERROR) << "Failed to create PoseLandmarker: " << status;
      return ToMpStatus(status);
    }

    PoseLandmarkerOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppPoseLandmarkerResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }
          PoseLandmarkerResult result;
          CppConvertToPoseLandmarkerResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppClosePoseLandmarkerResult(&result);
        };
  }

  auto landmarker = PoseLandmarker::Create(std::move(cpp_options));
  if (!landmarker.ok()) {
    ABSL_LOG(ERROR) << "Failed to create PoseLandmarker: "
                    << landmarker.status();
    *landmarker_out = nullptr;
    return ToMpStatus(landmarker.status());
  }
  *landmarker_out =
      new MpPoseLandmarkerInternal{.landmarker = std::move(*landmarker)};
  return kMpOk;
}

MpStatus CppPoseLandmarkerDetect(
    MpPoseLandmarkerPtr landmarker, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    PoseLandmarkerResult* result) {
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
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToPoseLandmarkerResult(*cpp_result, result);
  return kMpOk;
}

MpStatus CppPoseLandmarkerDetectForVideo(
    MpPoseLandmarkerPtr landmarker, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, PoseLandmarkerResult* result) {
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
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToPoseLandmarkerResult(*cpp_result, result);
  return kMpOk;
}

MpStatus CppPoseLandmarkerDetectAsync(
    MpPoseLandmarkerPtr landmarker, const MpImagePtr image,
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

void CppPoseLandmarkerCloseResult(PoseLandmarkerResult* result) {
  CppClosePoseLandmarkerResult(result);
}

MpStatus CppPoseLandmarkerClose(MpPoseLandmarkerPtr landmarker) {
  auto cpp_landmarker = landmarker->landmarker.get();
  auto result = cpp_landmarker->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close PoseLandmarker: " << result;
    return ToMpStatus(result);
  }
  delete landmarker;
  return kMpOk;
}

}  // namespace mediapipe::tasks::c::vision::pose_landmarker

extern "C" {

MpStatus MpPoseLandmarkerCreate(struct PoseLandmarkerOptions* options,
                                MpPoseLandmarkerPtr* landmarker_out) {
  return mediapipe::tasks::c::vision::pose_landmarker::CppPoseLandmarkerCreate(
      *options, landmarker_out);
}

MpStatus MpPoseLandmarkerDetectImage(MpPoseLandmarkerPtr landmarker,
                                     const MpImagePtr image,
                                     const ImageProcessingOptions* options,
                                     PoseLandmarkerResult* result) {
  return mediapipe::tasks::c::vision::pose_landmarker::CppPoseLandmarkerDetect(
      landmarker, image, options, result);
}

MpStatus MpPoseLandmarkerDetectForVideo(MpPoseLandmarkerPtr landmarker,
                                        const MpImagePtr image,
                                        const ImageProcessingOptions* options,
                                        int64_t timestamp_ms,
                                        PoseLandmarkerResult* result) {
  return mediapipe::tasks::c::vision::pose_landmarker::
      CppPoseLandmarkerDetectForVideo(landmarker, image, options, timestamp_ms,
                                      result);
}

MpStatus MpPoseLandmarkerDetectAsync(MpPoseLandmarkerPtr landmarker,
                                     const MpImagePtr image,
                                     const ImageProcessingOptions* options,
                                     int64_t timestamp_ms) {
  return mediapipe::tasks::c::vision::pose_landmarker::
      CppPoseLandmarkerDetectAsync(landmarker, image, options, timestamp_ms);
}

void MpPoseLandmarkerCloseResult(PoseLandmarkerResult* result) {
  mediapipe::tasks::c::vision::pose_landmarker::CppPoseLandmarkerCloseResult(
      result);
}

MpStatus MpPoseLandmarkerClose(MpPoseLandmarkerPtr landmarker) {
  return mediapipe::tasks::c::vision::pose_landmarker::CppPoseLandmarkerClose(
      landmarker);
}

}  // extern "C"
