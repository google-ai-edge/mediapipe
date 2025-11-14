/* Copyright 2024 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/vision/face_detector/face_detector.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/components/containers/detection_result_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/face_detector/face_detector.h"

struct MpFaceDetectorInternal {
  std::unique_ptr<::mediapipe::tasks::vision::face_detector::FaceDetector>
      detector;
};

namespace mediapipe::tasks::c::vision::face_detector {

namespace {

using ::mediapipe::tasks::c::components::containers::CppCloseDetectionResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToDetectionResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::face_detector::FaceDetector;
using CppFaceDetectorResult =
    ::mediapipe::tasks::vision::face_detector::FaceDetectorResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

}  // namespace

void CppConvertToFaceDetectorOptions(
    const FaceDetectorOptions& in,
    mediapipe::tasks::vision::face_detector::FaceDetectorOptions* out) {
  out->min_detection_confidence = in.min_detection_confidence;
  out->min_suppression_threshold = in.min_suppression_threshold;
}

MpStatus CppFaceDetectorCreate(const FaceDetectorOptions& options,
                               MpFaceDetectorPtr* detector) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::vision::face_detector::FaceDetectorOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToFaceDetectorOptions(options, cpp_options.get());
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  // Enable callback for processing live stream data when the running mode is
  // set to RunningMode::LIVE_STREAM.
  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      const absl::Status status = absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
      ABSL_LOG(ERROR) << "Failed to create FaceDetector: " << status;
      return ToMpStatus(status);
    }

    FaceDetectorOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppFaceDetectorResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }
          FaceDetectorResult result;
          CppConvertToDetectionResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppCloseDetectionResult(&result);
        };
  }

  auto cpp_detector = FaceDetector::Create(std::move(cpp_options));
  if (!cpp_detector.ok()) {
    ABSL_LOG(ERROR) << "Failed to create FaceDetector: "
                    << cpp_detector.status();
    return ToMpStatus(cpp_detector.status());
  }
  *detector = new MpFaceDetectorInternal{.detector = std::move(*cpp_detector)};
  return kMpOk;
}

MpStatus CppFaceDetectorDetect(
    MpFaceDetectorPtr detector, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    FaceDetectorResult* result) {
  auto cpp_detector = detector->detector.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result =
      cpp_detector->Detect(ToImage(image), cpp_image_processing_options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToDetectionResult(*cpp_result, result);
  return kMpOk;
}

MpStatus CppFaceDetectorDetectForVideo(
    MpFaceDetectorPtr detector, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, FaceDetectorResult* result) {
  auto cpp_detector = detector->detector.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result = cpp_detector->DetectForVideo(ToImage(image), timestamp_ms,
                                                 cpp_image_processing_options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToDetectionResult(*cpp_result, result);
  return kMpOk;
}

MpStatus CppFaceDetectorDetectAsync(
    MpFaceDetectorPtr detector, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  auto cpp_detector = detector->detector.get();
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  auto cpp_result = cpp_detector->DetectAsync(ToImage(image), timestamp_ms,
                                              cpp_image_processing_options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Data preparation for the landmark detection failed: "
                    << cpp_result;
    return ToMpStatus(cpp_result);
  }
  return kMpOk;
}

void CppFaceDetectorCloseResult(FaceDetectorResult* result) {
  CppCloseDetectionResult(result);
}

MpStatus CppFaceDetectorClose(MpFaceDetectorPtr detector) {
  auto cpp_detector = detector->detector.get();
  auto result = cpp_detector->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close FaceDetector: " << result;
    return ToMpStatus(result);
  }
  delete detector;
  return kMpOk;
}

}  // namespace mediapipe::tasks::c::vision::face_detector

extern "C" {

MP_EXPORT MpStatus MpFaceDetectorCreate(struct FaceDetectorOptions* options,
                                        MpFaceDetectorPtr* detector) {
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorCreate(
      *options, detector);
}

MP_EXPORT MpStatus MpFaceDetectorDetectImage(
    MpFaceDetectorPtr detector, const MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    FaceDetectorResult* result) {
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorDetect(
      detector, image, image_processing_options, result);
}

MP_EXPORT MpStatus MpFaceDetectorDetectForVideo(
    MpFaceDetectorPtr detector, const MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, FaceDetectorResult* result) {
  return mediapipe::tasks::c::vision::face_detector::
      CppFaceDetectorDetectForVideo(detector, image, image_processing_options,
                                    timestamp_ms, result);
}

MP_EXPORT MpStatus MpFaceDetectorDetectAsync(
    MpFaceDetectorPtr detector, const MpImagePtr image,
    const struct ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorDetectAsync(
      detector, image, image_processing_options, timestamp_ms);
}

MP_EXPORT void MpFaceDetectorCloseResult(FaceDetectorResult* result) {
  mediapipe::tasks::c::vision::face_detector::CppFaceDetectorCloseResult(
      result);
}

MP_EXPORT MpStatus MpFaceDetectorClose(MpFaceDetectorPtr detector) {
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorClose(
      detector);
}

}  // extern "C"
