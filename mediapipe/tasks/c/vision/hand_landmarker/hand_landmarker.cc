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
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/vision/core/common.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result_converter.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe::tasks::c::vision::hand_landmarker {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseHandLandmarkerResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToHandLandmarkerResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::vision::CreateImageFromBuffer;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::hand_landmarker::HandLandmarker;
typedef ::mediapipe::tasks::vision::hand_landmarker::HandLandmarkerResult
    CppHandLandmarkerResult;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
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

HandLandmarker* CppHandLandmarkerCreate(const HandLandmarkerOptions& options,
                                        char** error_msg) {
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
      CppProcessError(status, error_msg);
      return nullptr;
    }

    HandLandmarkerOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppHandLandmarkerResult> cpp_result,
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
          auto result = std::make_unique<HandLandmarkerResult>();
          CppConvertToHandLandmarkerResult(*cpp_result, result.get());

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

  auto landmarker = HandLandmarker::Create(std::move(cpp_options));
  if (!landmarker.ok()) {
    ABSL_LOG(ERROR) << "Failed to create HandLandmarker: "
                    << landmarker.status();
    CppProcessError(landmarker.status(), error_msg);
    return nullptr;
  }
  return landmarker->release();
}

int CppHandLandmarkerDetect(void* landmarker, const MpImage* image,
                            HandLandmarkerResult* result, char** error_msg) {
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

  auto cpp_landmarker = static_cast<HandLandmarker*>(landmarker);
  auto cpp_result = cpp_landmarker->Detect(*img);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Recognition failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToHandLandmarkerResult(*cpp_result, result);
  return 0;
}

int CppHandLandmarkerDetectForVideo(void* landmarker, const MpImage* image,
                                    int64_t timestamp_ms,
                                    HandLandmarkerResult* result,
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

  auto cpp_landmarker = static_cast<HandLandmarker*>(landmarker);
  auto cpp_result = cpp_landmarker->DetectForVideo(*img, timestamp_ms);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Recognition failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToHandLandmarkerResult(*cpp_result, result);
  return 0;
}

int CppHandLandmarkerDetectAsync(void* landmarker, const MpImage* image,
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

  auto cpp_landmarker = static_cast<HandLandmarker*>(landmarker);
  auto cpp_result = cpp_landmarker->DetectAsync(*img, timestamp_ms);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Data preparation for the landmark detection failed: "
                    << cpp_result;
    return CppProcessError(cpp_result, error_msg);
  }
  return 0;
}

void CppHandLandmarkerCloseResult(HandLandmarkerResult* result) {
  CppCloseHandLandmarkerResult(result);
}

int CppHandLandmarkerClose(void* landmarker, char** error_msg) {
  auto cpp_landmarker = static_cast<HandLandmarker*>(landmarker);
  auto result = cpp_landmarker->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close HandLandmarker: " << result;
    return CppProcessError(result, error_msg);
  }
  delete cpp_landmarker;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::hand_landmarker

extern "C" {

void* hand_landmarker_create(struct HandLandmarkerOptions* options,
                             char** error_msg) {
  return mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerCreate(
      *options, error_msg);
}

int hand_landmarker_detect_image(void* landmarker, const MpImage* image,
                                 HandLandmarkerResult* result,
                                 char** error_msg) {
  return mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerDetect(
      landmarker, image, result, error_msg);
}

int hand_landmarker_detect_for_video(void* landmarker, const MpImage* image,
                                     int64_t timestamp_ms,
                                     HandLandmarkerResult* result,
                                     char** error_msg) {
  return mediapipe::tasks::c::vision::hand_landmarker::
      CppHandLandmarkerDetectForVideo(landmarker, image, timestamp_ms, result,
                                      error_msg);
}

int hand_landmarker_detect_async(void* landmarker, const MpImage* image,
                                 int64_t timestamp_ms, char** error_msg) {
  return mediapipe::tasks::c::vision::hand_landmarker::
      CppHandLandmarkerDetectAsync(landmarker, image, timestamp_ms, error_msg);
}

void hand_landmarker_close_result(HandLandmarkerResult* result) {
  mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerCloseResult(
      result);
}

int hand_landmarker_close(void* landmarker, char** error_ms) {
  return mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerClose(
      landmarker, error_ms);
}

}  // extern "C"
