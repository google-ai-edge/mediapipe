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
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::face_detector::FaceDetector;
typedef ::mediapipe::tasks::vision::face_detector::FaceDetectorResult
    CppFaceDetectorResult;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
}

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

}  // namespace

void CppConvertToFaceDetectorOptions(
    const FaceDetectorOptions& in,
    mediapipe::tasks::vision::face_detector::FaceDetectorOptions* out) {
  out->min_detection_confidence = in.min_detection_confidence;
  out->min_suppression_threshold = in.min_suppression_threshold;
}

MpFaceDetectorPtr CppFaceDetectorCreate(const FaceDetectorOptions& options,
                                        char** error_msg) {
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
      CppProcessError(status, error_msg);
      return nullptr;
    }

    FaceDetectorOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppFaceDetectorResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          char* error_msg = nullptr;

          if (!cpp_result.ok()) {
            ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
            CppProcessError(cpp_result.status(), &error_msg);
            result_callback(nullptr, nullptr, timestamp, error_msg);
            free(error_msg);
            return;
          }

          // Result is valid for the lifetime of the callback function.
          auto result = std::make_unique<FaceDetectorResult>();
          CppConvertToDetectionResult(*cpp_result, result.get());

          MpImageInternal mp_image = {.image = image};
          result_callback(result.release(), &mp_image, timestamp,
                          /* error_msg= */ nullptr);
        };
  }

  auto detector = FaceDetector::Create(std::move(cpp_options));
  if (!detector.ok()) {
    ABSL_LOG(ERROR) << "Failed to create FaceDetector: " << detector.status();
    CppProcessError(detector.status(), error_msg);
    return nullptr;
  }
  return new MpFaceDetectorInternal{.detector = std::move(*detector)};
}

int CppFaceDetectorDetect(
    MpFaceDetectorPtr detector, const MpImagePtr image,
    std::optional<mediapipe::tasks::vision::core::ImageProcessingOptions>
        options,
    FaceDetectorResult* result, char** error_msg) {
  auto cpp_detector = detector->detector.get();
  auto cpp_result = cpp_detector->Detect(ToImage(image), options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToDetectionResult(*cpp_result, result);
  return 0;
}

int CppFaceDetectorDetectForVideo(
    MpFaceDetectorPtr detector, const MpImagePtr image,
    std::optional<mediapipe::tasks::vision::core::ImageProcessingOptions>
        options,
    int64_t timestamp_ms, FaceDetectorResult* result, char** error_msg) {
  auto cpp_detector = detector->detector.get();
  auto cpp_result =
      cpp_detector->DetectForVideo(ToImage(image), timestamp_ms, options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToDetectionResult(*cpp_result, result);
  return 0;
}

int CppFaceDetectorDetectAsync(
    MpFaceDetectorPtr detector, const MpImagePtr image,
    std::optional<mediapipe::tasks::vision::core::ImageProcessingOptions>
        options,
    int64_t timestamp_ms, char** error_msg) {
  auto cpp_detector = detector->detector.get();
  auto cpp_result =
      cpp_detector->DetectAsync(ToImage(image), timestamp_ms, options);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Data preparation for the landmark detection failed: "
                    << cpp_result;
    return CppProcessError(cpp_result, error_msg);
  }
  return 0;
}

void CppFaceDetectorCloseResult(FaceDetectorResult* result) {
  CppCloseDetectionResult(result);
}

int CppFaceDetectorClose(MpFaceDetectorPtr detector, char** error_msg) {
  auto cpp_detector = detector->detector.get();
  auto result = cpp_detector->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close FaceDetector: " << result;
    return CppProcessError(result, error_msg);
  }
  delete detector;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::face_detector

extern "C" {

MP_EXPORT MpFaceDetectorPtr
face_detector_create(struct FaceDetectorOptions* options, char** error_msg) {
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorCreate(
      *options, error_msg);
}

MP_EXPORT int face_detector_detect_image(MpFaceDetectorPtr detector,
                                         const MpImagePtr image,
                                         FaceDetectorResult* result,
                                         char** error_msg) {
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorDetect(
      detector, image, /* options= */ std::nullopt, result, error_msg);
}

MP_EXPORT int face_detector_detect_image_with_options(
    MpFaceDetectorPtr detector, const MpImagePtr image,
    struct ImageProcessingOptions* options, FaceDetectorResult* result,
    char** error_msg) {
  mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
  mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions(
      *options, &cpp_options);
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorDetect(
      detector, image, std::move(cpp_options), result, error_msg);
}

MP_EXPORT int face_detector_detect_for_video(MpFaceDetectorPtr detector,
                                             const MpImagePtr image,
                                             int64_t timestamp_ms,
                                             FaceDetectorResult* result,
                                             char** error_msg) {
  return mediapipe::tasks::c::vision::face_detector::
      CppFaceDetectorDetectForVideo(detector, image,
                                    /* options= */ std::nullopt, timestamp_ms,
                                    result, error_msg);
}

MP_EXPORT int face_detector_detect_for_video_with_options(
    MpFaceDetectorPtr detector, const MpImagePtr image,
    struct ImageProcessingOptions* options, int64_t timestamp_ms,
    FaceDetectorResult* result, char** error_msg) {
  mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
  mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions(
      *options, &cpp_options);
  return mediapipe::tasks::c::vision::face_detector::
      CppFaceDetectorDetectForVideo(detector, image, std::move(cpp_options),
                                    timestamp_ms, result, error_msg);
}

MP_EXPORT int face_detector_detect_async(MpFaceDetectorPtr detector,
                                         const MpImagePtr image,
                                         int64_t timestamp_ms,
                                         char** error_msg) {
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorDetectAsync(
      detector, image, /* options= */ std::nullopt, timestamp_ms, error_msg);
}

MP_EXPORT int face_detector_detect_async_with_options(
    MpFaceDetectorPtr detector, const MpImagePtr image,
    struct ImageProcessingOptions* options, int64_t timestamp_ms,
    char** error_msg) {
  mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
  mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions(
      *options, &cpp_options);
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorDetectAsync(
      detector, image, std::move(cpp_options), timestamp_ms, error_msg);
}

MP_EXPORT void face_detector_close_result(FaceDetectorResult* result) {
  mediapipe::tasks::c::vision::face_detector::CppFaceDetectorCloseResult(
      result);
}

MP_EXPORT int face_detector_close(MpFaceDetectorPtr detector, char** error_ms) {
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorClose(
      detector, error_ms);
}

}  // extern "C"
