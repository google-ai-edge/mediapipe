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
#include "mediapipe/tasks/c/core/base_options_converter.h"
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
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
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

MpHandLandmarkerPtr CppHandLandmarkerCreate(
    const HandLandmarkerOptions& options, char** error_msg) {
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
          MpImageInternal mp_image = {.image = image};
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
  return new MpHandLandmarkerInternal{.landmarker = std::move(*landmarker)};
}

int CppHandLandmarkerDetect(MpHandLandmarkerPtr landmarker, MpImagePtr image,
                            const ImageProcessingOptions* options,
                            HandLandmarkerResult* result, char** error_msg) {
  auto cpp_landmarker = landmarker->landmarker.get();
  absl::StatusOr<CppHandLandmarkerResult> cpp_result;

  if (options) {
    ::mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
    CppConvertToImageProcessingOptions(*options, &cpp_options);
    cpp_result = cpp_landmarker->Detect(ToImage(image), cpp_options);
  } else {
    cpp_result = cpp_landmarker->Detect(ToImage(image));
  }

  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Recognition failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToHandLandmarkerResult(*cpp_result, result);
  return 0;
}

int CppHandLandmarkerDetectForVideo(MpHandLandmarkerPtr landmarker,
                                    MpImagePtr image,
                                    const ImageProcessingOptions* options,
                                    int64_t timestamp_ms,
                                    HandLandmarkerResult* result,
                                    char** error_msg) {
  auto cpp_landmarker = landmarker->landmarker.get();
  absl::StatusOr<CppHandLandmarkerResult> cpp_result;

  if (options) {
    ::mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
    CppConvertToImageProcessingOptions(*options, &cpp_options);
    cpp_result = cpp_landmarker->DetectForVideo(ToImage(image), timestamp_ms,
                                                cpp_options);
  } else {
    cpp_result = cpp_landmarker->DetectForVideo(ToImage(image), timestamp_ms);
  }

  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Recognition failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToHandLandmarkerResult(*cpp_result, result);
  return 0;
}

int CppHandLandmarkerDetectAsync(MpHandLandmarkerPtr landmarker,
                                 MpImagePtr image,
                                 const ImageProcessingOptions* options,
                                 int64_t timestamp_ms, char** error_msg) {
  auto cpp_landmarker = landmarker->landmarker.get();
  absl::Status cpp_result;

  if (options) {
    ::mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
    CppConvertToImageProcessingOptions(*options, &cpp_options);
    cpp_result =
        cpp_landmarker->DetectAsync(ToImage(image), timestamp_ms, cpp_options);
  } else {
    cpp_result = cpp_landmarker->DetectAsync(ToImage(image), timestamp_ms);
  }

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

int CppHandLandmarkerClose(MpHandLandmarkerPtr landmarker, char** error_msg) {
  auto cpp_landmarker = landmarker->landmarker.get();
  auto result = cpp_landmarker->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close HandLandmarker: " << result;
    return CppProcessError(result, error_msg);
  }
  delete landmarker;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::hand_landmarker

extern "C" {

MP_EXPORT MpHandLandmarkerPtr hand_landmarker_create(
    struct HandLandmarkerOptions* options, char** error_msg) {
  return mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerCreate(
      *options, error_msg);
}

MP_EXPORT int hand_landmarker_detect_image(MpHandLandmarkerPtr landmarker,
                                           MpImagePtr image,
                                           HandLandmarkerResult* result,
                                           char** error_msg) {
  return mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerDetect(
      landmarker, image, /* options= */ nullptr, result, error_msg);
}

MP_EXPORT int hand_landmarker_detect_image_with_options(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    struct ImageProcessingOptions* options, HandLandmarkerResult* result,
    char** error_msg) {
  return mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerDetect(
      landmarker, image, options, result, error_msg);
}

MP_EXPORT int hand_landmarker_detect_for_video(MpHandLandmarkerPtr landmarker,
                                               MpImagePtr image,
                                               int64_t timestamp_ms,
                                               HandLandmarkerResult* result,
                                               char** error_msg) {
  return mediapipe::tasks::c::vision::hand_landmarker::
      CppHandLandmarkerDetectForVideo(landmarker, image, /* options= */ nullptr,
                                      timestamp_ms, result, error_msg);
}

MP_EXPORT int hand_landmarker_detect_for_video_with_options(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    struct ImageProcessingOptions* options, int64_t timestamp_ms,
    HandLandmarkerResult* result, char** error_msg) {
  return mediapipe::tasks::c::vision::hand_landmarker::
      CppHandLandmarkerDetectForVideo(landmarker, image, options, timestamp_ms,
                                      result, error_msg);
}

MP_EXPORT int hand_landmarker_detect_async(MpHandLandmarkerPtr landmarker,
                                           MpImagePtr image,
                                           int64_t timestamp_ms,
                                           char** error_msg) {
  return mediapipe::tasks::c::vision::hand_landmarker::
      CppHandLandmarkerDetectAsync(landmarker, image, /* options= */ nullptr,
                                   timestamp_ms, error_msg);
}

MP_EXPORT int hand_landmarker_detect_async_with_options(
    MpHandLandmarkerPtr landmarker, MpImagePtr image,
    struct ImageProcessingOptions* options, int64_t timestamp_ms,
    char** error_msg) {
  return mediapipe::tasks::c::vision::hand_landmarker::
      CppHandLandmarkerDetectAsync(landmarker, image, options, timestamp_ms,
                                   error_msg);
}

MP_EXPORT void hand_landmarker_close_result(HandLandmarkerResult* result) {
  mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerCloseResult(
      result);
}

MP_EXPORT int hand_landmarker_close(MpHandLandmarkerPtr landmarker,
                                    char** error_ms) {
  return mediapipe::tasks::c::vision::hand_landmarker::CppHandLandmarkerClose(
      landmarker, error_ms);
}

}  // extern "C"
