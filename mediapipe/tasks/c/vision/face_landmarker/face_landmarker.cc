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
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
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

struct MpFaceLandmarkerInternal {
  std::unique_ptr<::mediapipe::tasks::vision::face_landmarker::FaceLandmarker>
      landmarker;
};

namespace mediapipe::tasks::c::vision::face_landmarker {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseFaceLandmarkerResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToFaceLandmarkerResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::CreateImageFromBuffer;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::face_landmarker::FaceLandmarker;
typedef ::mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult
    CppFaceLandmarkerResult;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
}

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

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

MpFaceLandmarkerPtr CppFaceLandmarkerCreate(
    const FaceLandmarkerOptions& options, char** error_msg) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToFaceLandmarkerOptions(options, cpp_options.get());
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  // Enable callback for processing live stream data when the running mode is
  // set to RunningMode::LIVE_STREAM.
  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      const absl::Status status = absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
      ABSL_LOG(ERROR) << "Failed to create FaceLandmarker: " << status;
      CppProcessError(status, error_msg);
      return nullptr;
    }

    FaceLandmarkerOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppFaceLandmarkerResult> cpp_result,
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
          auto result = std::make_unique<FaceLandmarkerResult>();
          CppConvertToFaceLandmarkerResult(*cpp_result, result.get());
          MpImageInternal mp_image = {.image = image};
          result_callback(result.release(), &mp_image, timestamp,
                          /* error_msg= */ nullptr);
        };
  }

  auto landmarker = FaceLandmarker::Create(std::move(cpp_options));
  if (!landmarker.ok()) {
    ABSL_LOG(ERROR) << "Failed to create FaceLandmarker: "
                    << landmarker.status();
    CppProcessError(landmarker.status(), error_msg);
    return nullptr;
  }
  return new MpFaceLandmarkerInternal{.landmarker = std::move(*landmarker)};
}

int CppFaceLandmarkerDetect(MpFaceLandmarkerPtr landmarker, MpImagePtr image,
                            const ImageProcessingOptions* options,
                            FaceLandmarkerResult* result, char** error_msg) {
  auto cpp_landmarker = landmarker->landmarker.get();
  absl::StatusOr<CppFaceLandmarkerResult> cpp_result;

  if (options) {
    ::mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
    CppConvertToImageProcessingOptions(*options, &cpp_options);
    cpp_result = cpp_landmarker->Detect(ToImage(image), cpp_options);
  } else {
    cpp_result = cpp_landmarker->Detect(ToImage(image));
  }

  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToFaceLandmarkerResult(*cpp_result, result);
  return 0;
}

int CppFaceLandmarkerDetectForVideo(MpFaceLandmarkerPtr landmarker,
                                    MpImagePtr image,
                                    const ImageProcessingOptions* options,
                                    int64_t timestamp_ms,
                                    FaceLandmarkerResult* result,
                                    char** error_msg) {
  auto cpp_landmarker = landmarker->landmarker.get();
  absl::StatusOr<CppFaceLandmarkerResult> cpp_result;

  if (options) {
    ::mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
    CppConvertToImageProcessingOptions(*options, &cpp_options);
    cpp_result = cpp_landmarker->DetectForVideo(ToImage(image), timestamp_ms,
                                                cpp_options);
  } else {
    cpp_result = cpp_landmarker->DetectForVideo(ToImage(image), timestamp_ms);
  }

  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToFaceLandmarkerResult(*cpp_result, result);
  return 0;
}

int CppFaceLandmarkerDetectAsync(MpFaceLandmarkerPtr landmarker,
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

void CppFaceLandmarkerCloseResult(FaceLandmarkerResult* result) {
  CppCloseFaceLandmarkerResult(result);
}

int CppFaceLandmarkerClose(MpFaceLandmarkerPtr landmarker, char** error_msg) {
  auto cpp_landmarker = landmarker->landmarker.get();
  auto result = cpp_landmarker->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close FaceLandmarker: " << result;
    return CppProcessError(result, error_msg);
  }
  delete landmarker;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::face_landmarker

extern "C" {

MP_EXPORT MpFaceLandmarkerPtr face_landmarker_create(
    struct FaceLandmarkerOptions* options, char** error_msg) {
  return mediapipe::tasks::c::vision::face_landmarker::CppFaceLandmarkerCreate(
      *options, error_msg);
}

MP_EXPORT int face_landmarker_detect_image(MpFaceLandmarkerPtr landmarker,
                                           MpImagePtr image,
                                           FaceLandmarkerResult* result,
                                           char** error_msg) {
  return mediapipe::tasks::c::vision::face_landmarker::CppFaceLandmarkerDetect(
      landmarker, image, /* options= */ nullptr, result, error_msg);
}

MP_EXPORT int face_landmarker_detect_image_with_options(
    MpFaceLandmarkerPtr landmarker, MpImagePtr image,
    struct ImageProcessingOptions* options, FaceLandmarkerResult* result,
    char** error_msg) {
  return mediapipe::tasks::c::vision::face_landmarker::CppFaceLandmarkerDetect(
      landmarker, image, options, result, error_msg);
}

MP_EXPORT int face_landmarker_detect_for_video(MpFaceLandmarkerPtr landmarker,
                                               MpImagePtr image,
                                               int64_t timestamp_ms,
                                               FaceLandmarkerResult* result,
                                               char** error_msg) {
  return mediapipe::tasks::c::vision::face_landmarker::
      CppFaceLandmarkerDetectForVideo(landmarker, image, /* options= */ nullptr,
                                      timestamp_ms, result, error_msg);
}

MP_EXPORT int face_landmarker_detect_for_video_with_options(
    MpFaceLandmarkerPtr landmarker, MpImagePtr image,
    struct ImageProcessingOptions* options, int64_t timestamp_ms,
    FaceLandmarkerResult* result, char** error_msg) {
  return mediapipe::tasks::c::vision::face_landmarker::
      CppFaceLandmarkerDetectForVideo(landmarker, image, options, timestamp_ms,
                                      result, error_msg);
}

MP_EXPORT int face_landmarker_detect_async(MpFaceLandmarkerPtr landmarker,
                                           MpImagePtr image,
                                           int64_t timestamp_ms,
                                           char** error_msg) {
  return mediapipe::tasks::c::vision::face_landmarker::
      CppFaceLandmarkerDetectAsync(landmarker, image, /* options= */ nullptr,
                                   timestamp_ms, error_msg);
}

MP_EXPORT int face_landmarker_detect_async_with_options(
    MpFaceLandmarkerPtr landmarker, MpImagePtr image,
    struct ImageProcessingOptions* options, int64_t timestamp_ms,
    char** error_msg) {
  return mediapipe::tasks::c::vision::face_landmarker::
      CppFaceLandmarkerDetectAsync(landmarker, image, options, timestamp_ms,
                                   error_msg);
}

MP_EXPORT void face_landmarker_close_result(FaceLandmarkerResult* result) {
  mediapipe::tasks::c::vision::face_landmarker::CppFaceLandmarkerCloseResult(
      result);
}

MP_EXPORT int face_landmarker_close(MpFaceLandmarkerPtr landmarker,
                                    char** error_ms) {
  return mediapipe::tasks::c::vision::face_landmarker::CppFaceLandmarkerClose(
      landmarker, error_ms);
}

}  // extern "C"
