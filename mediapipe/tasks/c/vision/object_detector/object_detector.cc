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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/components/containers/detection_result_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
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
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::ObjectDetector;
using ::mediapipe::tasks::vision::core::RunningMode;
using CppObjectDetectorResult =
    ::mediapipe::tasks::vision::ObjectDetectorResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;

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

MpStatus CppObjectDetectorCreate(const ObjectDetectorOptions& options,
                                 MpObjectDetectorPtr* detector_out) {
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
      *detector_out = nullptr;
      return ToMpStatus(status);
    }

    ObjectDetectorOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppObjectDetectorResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }
          ObjectDetectorResult result;
          CppConvertToDetectionResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppCloseDetectionResult(&result);
        };
  }

  auto detector = ObjectDetector::Create(std::move(cpp_options));
  if (!detector.ok()) {
    ABSL_LOG(ERROR) << "Failed to create ObjectDetector: " << detector.status();
    *detector_out = nullptr;
    return ToMpStatus(detector.status());
  }
  *detector_out =
      new MpObjectDetectorInternal{.detector = std::move(*detector)};
  return kMpOk;
}

MpStatus CppObjectDetectorDetect(
    MpObjectDetectorPtr detector, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    ObjectDetectorResult* result) {
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

MpStatus CppObjectDetectorDetectForVideo(
    MpObjectDetectorPtr detector, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, ObjectDetectorResult* result) {
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

MpStatus CppObjectDetectorDetectAsync(
    MpObjectDetectorPtr detector, const MpImagePtr image,
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
    ABSL_LOG(ERROR) << "Data preparation for the object detection failed: "
                    << cpp_result;
    return ToMpStatus(cpp_result);
  }
  return kMpOk;
}

void CppObjectDetectorCloseResult(ObjectDetectorResult* result) {
  CppCloseDetectionResult(result);
}

MpStatus CppObjectDetectorClose(MpObjectDetectorPtr detector) {
  if (!detector) return kMpOk;
  auto cpp_detector = detector->detector.get();
  auto result = cpp_detector->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close ObjectDetector: " << result;
    return ToMpStatus(result);
  }
  delete detector;
  return kMpOk;
}

}  // namespace mediapipe::tasks::c::vision::object_detector

extern "C" {

MpStatus MpObjectDetectorCreate(struct ObjectDetectorOptions* options,
                                MpObjectDetectorPtr* detector_out) {
  return mediapipe::tasks::c::vision::object_detector::CppObjectDetectorCreate(
      *options, detector_out);
}

MpStatus MpObjectDetectorDetectImage(
    MpObjectDetectorPtr detector, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    ObjectDetectorResult* result) {
  return mediapipe::tasks::c::vision::object_detector::CppObjectDetectorDetect(
      detector, image, image_processing_options, result);
}

MpStatus MpObjectDetectorDetectForVideo(
    MpObjectDetectorPtr detector, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, ObjectDetectorResult* result) {
  return mediapipe::tasks::c::vision::object_detector::
      CppObjectDetectorDetectForVideo(detector, image, image_processing_options,
                                      timestamp_ms, result);
}

MpStatus MpObjectDetectorDetectAsync(
    MpObjectDetectorPtr detector, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  return mediapipe::tasks::c::vision::object_detector::
      CppObjectDetectorDetectAsync(detector, image, image_processing_options,
                                   timestamp_ms);
}

void MpObjectDetectorCloseResult(ObjectDetectorResult* result) {
  mediapipe::tasks::c::vision::object_detector::CppObjectDetectorCloseResult(
      result);
}

MpStatus MpObjectDetectorClose(MpObjectDetectorPtr detector) {
  return mediapipe::tasks::c::vision::object_detector::CppObjectDetectorClose(
      detector);
}

}  // extern "C"
