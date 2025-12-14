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

#include "mediapipe/tasks/c/vision/image_classifier/image_classifier.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/components/containers/classification_result_converter.h"
#include "mediapipe/tasks/c/components/processors/classifier_options_converter.h"
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
#include "mediapipe/tasks/cc/vision/image_classifier/image_classifier.h"

struct MpImageClassifierInternal {
  std::unique_ptr<::mediapipe::tasks::vision::image_classifier::ImageClassifier>
      instance;
};

namespace mediapipe::tasks::c::vision::image_classifier {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseClassificationResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToClassificationResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToClassifierOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::c::vision::core::CppConvertToImageProcessingOptions;
using ::mediapipe::tasks::vision::core::RunningMode;
using ::mediapipe::tasks::vision::image_classifier::ImageClassifier;
using CppImageClassifierResult =
    ::mediapipe::tasks::vision::image_classifier::ImageClassifierResult;
using CppImageProcessingOptions =
    ::mediapipe::tasks::vision::core::ImageProcessingOptions;

const Image& ToImage(const MpImagePtr mp_image) { return mp_image->image; }

ImageClassifier* GetCppClassifier(MpImageClassifierPtr classifier) {
  ABSL_CHECK(classifier != nullptr) << "ImageClassifier is null.";
  return classifier->instance.get();
}

}  // namespace

absl::Status CppMpImageClassifierCreate(const ImageClassifierOptions& options,
                                        MpImageClassifierPtr* classifier) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::vision::image_classifier::ImageClassifierOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToClassifierOptions(options.classifier_options,
                                &cpp_options->classifier_options);
  cpp_options->running_mode = static_cast<RunningMode>(options.running_mode);

  // Enable callback for processing live stream data when the running mode is
  // set to RunningMode::LIVE_STREAM.
  if (cpp_options->running_mode == RunningMode::LIVE_STREAM) {
    if (options.result_callback == nullptr) {
      return absl::InvalidArgumentError(
          "Provided null pointer to callback function.");
    }

    ImageClassifierOptions::result_callback_fn result_callback =
        options.result_callback;
    cpp_options->result_callback =
        [result_callback](absl::StatusOr<CppImageClassifierResult> cpp_result,
                          const Image& image, int64_t timestamp) {
          MpImageInternal mp_image({.image = image});
          if (!cpp_result.ok()) {
            result_callback(ToMpStatus(cpp_result.status()), nullptr, &mp_image,
                            timestamp);
            return;
          }

          ImageClassifierResult result;
          CppConvertToClassificationResult(*cpp_result, &result);
          result_callback(kMpOk, &result, &mp_image, timestamp);
          CppCloseClassificationResult(&result);
        };
  }

  auto cpp_classifier = ImageClassifier::Create(std::move(cpp_options));
  if (!cpp_classifier.ok()) {
    return cpp_classifier.status();
  }
  *classifier =
      new MpImageClassifierInternal{.instance = std::move(*cpp_classifier)};
  return absl::OkStatus();
}

absl::Status CppMpImageClassifierClassifyImage(
    MpImageClassifierPtr classifier, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    ImageClassifierResult* result) {
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  ImageClassifier* cpp_classifier = GetCppClassifier(classifier);
  auto cpp_result =
      cpp_classifier->Classify(ToImage(image), cpp_image_processing_options);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToClassificationResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppMpImageClassifierClassifyForVideo(
    MpImageClassifierPtr classifier, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, ImageClassifierResult* result) {
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  ImageClassifier* cpp_classifier = GetCppClassifier(classifier);
  auto cpp_result = cpp_classifier->ClassifyForVideo(
      ToImage(image), timestamp_ms, cpp_image_processing_options);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToClassificationResult(*cpp_result, result);
  return absl::OkStatus();
}

absl::Status CppMpImageClassifierClassifyAsync(
    MpImageClassifierPtr classifier, const MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms) {
  std::optional<CppImageProcessingOptions> cpp_image_processing_options;
  if (image_processing_options) {
    CppImageProcessingOptions options;
    CppConvertToImageProcessingOptions(*image_processing_options, &options);
    cpp_image_processing_options = options;
  }
  ImageClassifier* cpp_classifier = GetCppClassifier(classifier);
  return cpp_classifier->ClassifyAsync(ToImage(image), timestamp_ms,
                                       cpp_image_processing_options);
}

void CppMpImageClassifierCloseResult(ImageClassifierResult* result) {
  CppCloseClassificationResult(result);
}

absl::Status CppMpImageClassifierClose(MpImageClassifierPtr classifier) {
  ImageClassifier* cpp_classifier = GetCppClassifier(classifier);
  auto result = cpp_classifier->Close();
  if (!result.ok()) {
    return result;
  }
  delete classifier;
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::vision::image_classifier

extern "C" {

MP_EXPORT MpStatus
MpImageClassifierCreate(struct ImageClassifierOptions* options,
                        MpImageClassifierPtr* classifier, char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::vision::image_classifier::CppMpImageClassifierCreate(
          *options, classifier);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpImageClassifierClassifyImage(
    MpImageClassifierPtr classifier, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    ImageClassifierResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::image_classifier::
      CppMpImageClassifierClassifyImage(classifier, image,
                                        image_processing_options, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpImageClassifierClassifyForVideo(
    MpImageClassifierPtr classifier, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, ImageClassifierResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::image_classifier::
      CppMpImageClassifierClassifyForVideo(
          classifier, image, image_processing_options, timestamp_ms, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpImageClassifierClassifyAsync(
    MpImageClassifierPtr classifier, MpImagePtr image,
    const ImageProcessingOptions* image_processing_options,
    int64_t timestamp_ms, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::vision::image_classifier::
      CppMpImageClassifierClassifyAsync(classifier, image,
                                        image_processing_options, timestamp_ms);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpImageClassifierCloseResult(ImageClassifierResult* result) {
  mediapipe::tasks::c::vision::image_classifier::
      CppMpImageClassifierCloseResult(result);
}

MP_EXPORT MpStatus MpImageClassifierClose(MpImageClassifierPtr classifier,
                                          char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::vision::image_classifier::CppMpImageClassifierClose(
          classifier);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
