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
#include "mediapipe/tasks/c/components/containers/detection_result_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options_converter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/object_detector/object_detector.h"

struct MpObjectDetectorInternal {
  std::unique_ptr<::mediapipe::tasks::vision::ObjectDetector> detector;
};

namespace mediapipe::tasks::c::vision::object_detector {

namespace {

using ::mediapipe::tasks::c::components::containers::CppCloseDetectionResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToDetectionResult;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
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

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

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

MpObjectDetectorPtr CppObjectDetectorCreate(
    const ObjectDetectorOptions& options, char** error_msg) {
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
            result_callback(nullptr, nullptr, timestamp, error_msg);
            free(error_msg);
            return;
          }

          // Result is valid for the lifetime of the callback function.
          ObjectDetectorResult result;
          CppConvertToDetectionResult(*cpp_result, &result);

          MpImageInternal mp_image = {.image = image};

          result_callback(&result, &mp_image, timestamp,
                          /* error_msg= */ nullptr);
        };
  }

  auto detector = ObjectDetector::Create(std::move(cpp_options));
  if (!detector.ok()) {
    ABSL_LOG(ERROR) << "Failed to create ObjectDetector: " << detector.status();
    CppProcessError(detector.status(), error_msg);
    return nullptr;
  }
  return new MpObjectDetectorInternal{.detector = std::move(*detector)};
}

int CppObjectDetectorDetect(MpObjectDetectorPtr detector,
                            const MpImagePtr image,
                            const ImageProcessingOptions* options,
                            ObjectDetectorResult* result, char** error_msg) {
  auto cpp_detector = detector->detector.get();
  absl::StatusOr<CppObjectDetectorResult> cpp_result;

  if (options) {
    ::mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
    CppConvertToImageProcessingOptions(*options, &cpp_options);
    cpp_result = cpp_detector->Detect(ToImage(image), cpp_options);
  } else {
    cpp_result = cpp_detector->Detect(ToImage(image));
  }

  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToDetectionResult(*cpp_result, result);
  return 0;
}

int CppObjectDetectorDetectForVideo(MpObjectDetectorPtr detector,
                                    const MpImagePtr image,
                                    const ImageProcessingOptions* options,
                                    int64_t timestamp_ms,
                                    ObjectDetectorResult* result,
                                    char** error_msg) {
  auto cpp_detector = detector->detector.get();
  absl::StatusOr<CppObjectDetectorResult> cpp_result;

  if (options) {
    ::mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
    CppConvertToImageProcessingOptions(*options, &cpp_options);
    cpp_result =
        cpp_detector->DetectForVideo(ToImage(image), timestamp_ms, cpp_options);
  } else {
    cpp_result = cpp_detector->DetectForVideo(ToImage(image), timestamp_ms);
  }

  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Detection failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToDetectionResult(*cpp_result, result);
  return 0;
}

int CppObjectDetectorDetectAsync(MpObjectDetectorPtr detector,
                                 const MpImagePtr image,
                                 const ImageProcessingOptions* options,
                                 int64_t timestamp_ms, char** error_msg) {
  auto cpp_detector = detector->detector.get();
  absl::Status cpp_result;

  if (options) {
    ::mediapipe::tasks::vision::core::ImageProcessingOptions cpp_options;
    CppConvertToImageProcessingOptions(*options, &cpp_options);
    cpp_result =
        cpp_detector->DetectAsync(ToImage(image), timestamp_ms, cpp_options);
  } else {
    cpp_result = cpp_detector->DetectAsync(ToImage(image), timestamp_ms);
  }

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

int CppObjectDetectorClose(MpObjectDetectorPtr detector, char** error_msg) {
  auto cpp_detector = detector->detector.get();
  auto result = cpp_detector->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close ObjectDetector: " << result;
    return CppProcessError(result, error_msg);
  }
  delete detector;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::object_detector

extern "C" {

MP_EXPORT MpObjectDetectorPtr object_detector_create(
    struct ObjectDetectorOptions* options, char** error_msg) {
  return mediapipe::tasks::c::vision::object_detector::CppObjectDetectorCreate(
      *options, error_msg);
}

MP_EXPORT int object_detector_detect_image(MpObjectDetectorPtr detector,
                                           const MpImagePtr image,
                                           ObjectDetectorResult* result,
                                           char** error_msg) {
  return mediapipe::tasks::c::vision::object_detector::CppObjectDetectorDetect(
      detector, image, /*options=*/nullptr, result, error_msg);
}

MP_EXPORT int object_detector_detect_image_with_options(
    MpObjectDetectorPtr detector, const MpImagePtr image,
    const ImageProcessingOptions* options, ObjectDetectorResult* result,
    char** error_msg) {
  return mediapipe::tasks::c::vision::object_detector::CppObjectDetectorDetect(
      detector, image, options, result, error_msg);
}

MP_EXPORT int object_detector_detect_for_video(MpObjectDetectorPtr detector,
                                               const MpImagePtr image,
                                               int64_t timestamp_ms,
                                               ObjectDetectorResult* result,
                                               char** error_msg) {
  return mediapipe::tasks::c::vision::object_detector::
      CppObjectDetectorDetectForVideo(detector, image, /* options= */ nullptr,
                                      timestamp_ms, result, error_msg);
}

MP_EXPORT int object_detector_detect_for_video_with_options(
    MpObjectDetectorPtr detector, const MpImagePtr image,
    const ImageProcessingOptions* options, int64_t timestamp_ms,
    ObjectDetectorResult* result, char** error_msg) {
  return mediapipe::tasks::c::vision::object_detector::
      CppObjectDetectorDetectForVideo(detector, image, options, timestamp_ms,
                                      result, error_msg);
}

MP_EXPORT int object_detector_detect_async(MpObjectDetectorPtr detector,
                                           const MpImagePtr image,
                                           int64_t timestamp_ms,
                                           char** error_msg) {
  return mediapipe::tasks::c::vision::object_detector::
      CppObjectDetectorDetectAsync(detector, image, /*options=*/nullptr,
                                   timestamp_ms, error_msg);
}

MP_EXPORT int object_detector_detect_async_with_options(
    MpObjectDetectorPtr detector, const MpImagePtr image,
    const ImageProcessingOptions* options, int64_t timestamp_ms,
    char** error_msg) {
  return mediapipe::tasks::c::vision::object_detector::
      CppObjectDetectorDetectAsync(detector, image, options, timestamp_ms,
                                   error_msg);
}

MP_EXPORT void object_detector_close_result(ObjectDetectorResult* result) {
  mediapipe::tasks::c::vision::object_detector::CppObjectDetectorCloseResult(
      result);
}

MP_EXPORT int object_detector_close(MpObjectDetectorPtr detector,
                                    char** error_ms) {
  return mediapipe::tasks::c::vision::object_detector::CppObjectDetectorClose(
      detector, error_ms);
}

}  // extern "C"
