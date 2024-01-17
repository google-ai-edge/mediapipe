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
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/c/components/containers/detection_result_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/face_detector/face_detector.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe::tasks::c::vision::face_detector {

namespace {

using ::mediapipe::tasks::c::components::containers::CppCloseDetectionResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToDetectionResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::vision::CreateImageFromBuffer;
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

}  // namespace

void CppConvertToFaceDetectorOptions(
    const FaceDetectorOptions& in,
    mediapipe::tasks::vision::face_detector::FaceDetectorOptions* out) {
  out->min_detection_confidence = in.min_detection_confidence;
  out->min_suppression_threshold = in.min_suppression_threshold;
}

FaceDetector* CppFaceDetectorCreate(const FaceDetectorOptions& options,
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
            result_callback({}, MpImage(), timestamp, error_msg);
            free(error_msg);
            return;
          }

          // Result is valid for the lifetime of the callback function.
          FaceDetectorResult result;
          CppConvertToDetectionResult(*cpp_result, &result);

          const auto& image_frame = image.GetImageFrameSharedPtr();
          const MpImage mp_image = {
              .type = MpImage::IMAGE_FRAME,
              .image_frame = {
                  .format = static_cast<::ImageFormat>(image_frame->Format()),
                  .image_buffer = image_frame->PixelData(),
                  .width = image_frame->Width(),
                  .height = image_frame->Height()}};

          result_callback(&result, mp_image, timestamp,
                          /* error_msg= */ nullptr);

          CppCloseDetectionResult(&result);
        };
  }

  auto detector = FaceDetector::Create(std::move(cpp_options));
  if (!detector.ok()) {
    ABSL_LOG(ERROR) << "Failed to create FaceDetector: " << detector.status();
    CppProcessError(detector.status(), error_msg);
    return nullptr;
  }
  return detector->release();
}

int CppFaceDetectorDetect(void* detector, const MpImage& image,
                          FaceDetectorResult* result, char** error_msg) {
  if (image.type == MpImage::GPU_BUFFER) {
    const absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet.");

    ABSL_LOG(ERROR) << "Detection failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image.image_frame.format),
      image.image_frame.image_buffer, image.image_frame.width,
      image.image_frame.height);

  if (!img.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_detector = static_cast<FaceDetector*>(detector);
  auto cpp_result = cpp_detector->Detect(*img);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToDetectionResult(*cpp_result, result);
  return 0;
}

int CppFaceDetectorDetectForVideo(void* detector, const MpImage& image,
                                  int64_t timestamp_ms,
                                  FaceDetectorResult* result,
                                  char** error_msg) {
  if (image.type == MpImage::GPU_BUFFER) {
    absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet");

    ABSL_LOG(ERROR) << "Detection failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image.image_frame.format),
      image.image_frame.image_buffer, image.image_frame.width,
      image.image_frame.height);

  if (!img.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_detector = static_cast<FaceDetector*>(detector);
  auto cpp_result = cpp_detector->DetectForVideo(*img, timestamp_ms);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToDetectionResult(*cpp_result, result);
  return 0;
}

int CppFaceDetectorDetectAsync(void* detector, const MpImage& image,
                               int64_t timestamp_ms, char** error_msg) {
  if (image.type == MpImage::GPU_BUFFER) {
    absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet");

    ABSL_LOG(ERROR) << "Detection failed: " << status.message();
    return CppProcessError(status, error_msg);
  }

  const auto img = CreateImageFromBuffer(
      static_cast<ImageFormat::Format>(image.image_frame.format),
      image.image_frame.image_buffer, image.image_frame.width,
      image.image_frame.height);

  if (!img.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Image: " << img.status();
    return CppProcessError(img.status(), error_msg);
  }

  auto cpp_detector = static_cast<FaceDetector*>(detector);
  auto cpp_result = cpp_detector->DetectAsync(*img, timestamp_ms);
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

int CppFaceDetectorClose(void* detector, char** error_msg) {
  auto cpp_detector = static_cast<FaceDetector*>(detector);
  auto result = cpp_detector->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close FaceDetector: " << result;
    return CppProcessError(result, error_msg);
  }
  delete cpp_detector;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::face_detector

extern "C" {

void* face_detector_create(struct FaceDetectorOptions* options,
                           char** error_msg) {
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorCreate(
      *options, error_msg);
}

int face_detector_detect_image(void* detector, const MpImage& image,
                               FaceDetectorResult* result, char** error_msg) {
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorDetect(
      detector, image, result, error_msg);
}

int face_detector_detect_for_video(void* detector, const MpImage& image,
                                   int64_t timestamp_ms,
                                   FaceDetectorResult* result,
                                   char** error_msg) {
  return mediapipe::tasks::c::vision::face_detector::
      CppFaceDetectorDetectForVideo(detector, image, timestamp_ms, result,
                                    error_msg);
}

int face_detector_detect_async(void* detector, const MpImage& image,
                               int64_t timestamp_ms, char** error_msg) {
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorDetectAsync(
      detector, image, timestamp_ms, error_msg);
}

void face_detector_close_result(FaceDetectorResult* result) {
  mediapipe::tasks::c::vision::face_detector::CppFaceDetectorCloseResult(
      result);
}

int face_detector_close(void* detector, char** error_ms) {
  return mediapipe::tasks::c::vision::face_detector::CppFaceDetectorClose(
      detector, error_ms);
}

}  // extern "C"
