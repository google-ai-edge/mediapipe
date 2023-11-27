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

#include "mediapipe/tasks/c/vision/object_detector/object_detector.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/c/components/containers/detection_result_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/object_detector/object_detector.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe::tasks::c::vision::object_detector {

namespace {

using ::mediapipe::tasks::c::components::containers::CppCloseDetectionResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToDetectionResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::vision::CreateImageFromBuffer;
using ::mediapipe::tasks::vision::ObjectDetector;
using ::mediapipe::tasks::vision::core::RunningMode;
typedef ::mediapipe::tasks::vision::ObjectDetectorResult
    CppObjectDetectorResult;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
}

}  // namespace

void CppConvertToDetectorOptions(
    const ObjectDetectorOptions& in,
    mediapipe::tasks::vision::ObjectDetectorOptions* out) {
  out->display_names_locale =
      in.display_names_locale ? std::string(in.display_names_locale) : "en";
  out->max_results = in.max_results;
  out->score_threshold = in.score_threshold;
  out->category_allowlist =
      std::vector<std::string>(in.category_allowlist_count);
  for (uint32_t i = 0; i < in.category_allowlist_count; ++i) {
    out->category_allowlist[i] = in.category_allowlist[i];
  }
  out->category_denylist = std::vector<std::string>(in.category_denylist_count);
  for (uint32_t i = 0; i < in.category_denylist_count; ++i) {
    out->category_denylist[i] = in.category_denylist[i];
  }
}

ObjectDetector* CppObjectDetectorCreate(const ObjectDetectorOptions& options,
                                        char** error_msg) {
  auto cpp_options =
      std::make_unique<::mediapipe::tasks::vision::ObjectDetectorOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToDetectorOptions(options, cpp_options.get());
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  // Enable callback for processing live stream data when the running mode is
  // set to RunningMode::LIVE_STREAM.
  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      const absl::Status status = absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
      ABSL_LOG(ERROR) << "Failed to create ObjectDetector: " << status;
      CppProcessError(status, error_msg);
      return nullptr;
    }

    ObjectDetectorOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppObjectDetectorResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          char* error_msg = nullptr;

          if (!cpp_result.ok()) {
            ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
            CppProcessError(cpp_result.status(), &error_msg);
            result_callback(nullptr, MpImage(), timestamp, error_msg);
            free(error_msg);
            return;
          }

          // Result is valid for the lifetime of the callback function.
          ObjectDetectorResult result;
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

  auto detector = ObjectDetector::Create(std::move(cpp_options));
  if (!detector.ok()) {
    ABSL_LOG(ERROR) << "Failed to create ObjectDetector: " << detector.status();
    CppProcessError(detector.status(), error_msg);
    return nullptr;
  }
  return detector->release();
}

int CppObjectDetectorDetect(void* detector, const MpImage* image,
                            ObjectDetectorResult* result, char** error_msg) {
  if (image->type == MpImage::GPU_BUFFER) {
    const absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet.");

    ABSL_LOG(ERROR) << "Detection failed: " << status.message();
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

  auto cpp_detector = static_cast<ObjectDetector*>(detector);
  auto cpp_result = cpp_detector->Detect(*img);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToDetectionResult(*cpp_result, result);
  return 0;
}

int CppObjectDetectorDetectForVideo(void* detector, const MpImage* image,
                                    int64_t timestamp_ms,
                                    ObjectDetectorResult* result,
                                    char** error_msg) {
  if (image->type == MpImage::GPU_BUFFER) {
    absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet");

    ABSL_LOG(ERROR) << "Detection failed: " << status.message();
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

  auto cpp_detector = static_cast<ObjectDetector*>(detector);
  auto cpp_result = cpp_detector->DetectForVideo(*img, timestamp_ms);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToDetectionResult(*cpp_result, result);
  return 0;
}

int CppObjectDetectorDetectAsync(void* detector, const MpImage* image,
                                 int64_t timestamp_ms, char** error_msg) {
  if (image->type == MpImage::GPU_BUFFER) {
    absl::Status status =
        absl::InvalidArgumentError("GPU Buffer not supported yet");

    ABSL_LOG(ERROR) << "Detection failed: " << status.message();
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

  auto cpp_detector = static_cast<ObjectDetector*>(detector);
  auto cpp_result = cpp_detector->DetectAsync(*img, timestamp_ms);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Data preparation for the object detection failed: "
                    << cpp_result;
    return CppProcessError(cpp_result, error_msg);
  }
  return 0;
}

void CppObjectDetectorCloseResult(ObjectDetectorResult* result) {
  CppCloseDetectionResult(result);
}

int CppObjectDetectorClose(void* detector, char** error_msg) {
  auto cpp_detector = static_cast<ObjectDetector*>(detector);
  auto result = cpp_detector->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close ObjectDetector: " << result;
    return CppProcessError(result, error_msg);
  }
  delete cpp_detector;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::object_detector

extern "C" {

void* object_detector_create(struct ObjectDetectorOptions* options,
                             char** error_msg) {
  return mediapipe::tasks::c::vision::object_detector::CppObjectDetectorCreate(
      *options, error_msg);
}

int object_detector_detect_image(void* detector, const MpImage* image,
                                 ObjectDetectorResult* result,
                                 char** error_msg) {
  return mediapipe::tasks::c::vision::object_detector::CppObjectDetectorDetect(
      detector, image, result, error_msg);
}

int object_detector_detect_for_video(void* detector, const MpImage* image,
                                     int64_t timestamp_ms,
                                     ObjectDetectorResult* result,
                                     char** error_msg) {
  return mediapipe::tasks::c::vision::object_detector::
      CppObjectDetectorDetectForVideo(detector, image, timestamp_ms, result,
                                      error_msg);
}

int object_detector_detect_async(void* detector, const MpImage* image,
                                 int64_t timestamp_ms, char** error_msg) {
  return mediapipe::tasks::c::vision::object_detector::
      CppObjectDetectorDetectAsync(detector, image, timestamp_ms, error_msg);
}

void object_detector_close_result(ObjectDetectorResult* result) {
  mediapipe::tasks::c::vision::object_detector::CppObjectDetectorCloseResult(
      result);
}

int object_detector_close(void* detector, char** error_ms) {
  return mediapipe::tasks::c::vision::object_detector::CppObjectDetectorClose(
      detector, error_ms);
}

}  // extern "C"
