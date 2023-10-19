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

#include <memory>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/c/components/containers/classification_result_converter.h"
#include "mediapipe/tasks/c/components/processors/classifier_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/cc/vision/image_classifier/image_classifier.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

namespace mediapipe::tasks::c::vision::image_classifier {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseClassificationResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToClassificationResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToClassifierOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::vision::CreateImageFromBuffer;
using ::mediapipe::tasks::vision::image_classifier::ImageClassifier;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
}

}  // namespace

ImageClassifier* CppImageClassifierCreate(const ImageClassifierOptions& options,
                                          char** error_msg) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::vision::image_classifier::ImageClassifierOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToClassifierOptions(options.classifier_options,
                                &cpp_options->classifier_options);

  auto classifier = ImageClassifier::Create(std::move(cpp_options));
  if (!classifier.ok()) {
    ABSL_LOG(ERROR) << "Failed to create ImageClassifier: "
                    << classifier.status();
    CppProcessError(classifier.status(), error_msg);
    return nullptr;
  }
  return classifier->release();
}

int CppImageClassifierClassify(void* classifier, const MpImage* image,
                               ImageClassifierResult* result,
                               char** error_msg) {
  if (image->type == MpImage::GPU_BUFFER) {
    absl::Status status =
        absl::InvalidArgumentError("gpu buffer not supported yet");

    ABSL_LOG(ERROR) << "Classification failed: " << status.message();
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

  auto cpp_classifier = static_cast<ImageClassifier*>(classifier);
  auto cpp_result = cpp_classifier->Classify(*img);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Classification failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToClassificationResult(*cpp_result, result);
  return 0;
}

void CppImageClassifierCloseResult(ImageClassifierResult* result) {
  CppCloseClassificationResult(result);
}

int CppImageClassifierClose(void* classifier, char** error_msg) {
  auto cpp_classifier = static_cast<ImageClassifier*>(classifier);
  auto result = cpp_classifier->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close ImageClassifier: " << result;
    return CppProcessError(result, error_msg);
  }
  delete cpp_classifier;
  return 0;
}

}  // namespace mediapipe::tasks::c::vision::image_classifier

extern "C" {

void* image_classifier_create(struct ImageClassifierOptions* options,
                              char** error_msg) {
  return mediapipe::tasks::c::vision::image_classifier::
      CppImageClassifierCreate(*options, error_msg);
}

int image_classifier_classify_image(void* classifier, const MpImage* image,
                                    ImageClassifierResult* result,
                                    char** error_msg) {
  return mediapipe::tasks::c::vision::image_classifier::
      CppImageClassifierClassify(classifier, image, result, error_msg);
}

void image_classifier_close_result(ImageClassifierResult* result) {
  mediapipe::tasks::c::vision::image_classifier::CppImageClassifierCloseResult(
      result);
}

int image_classifier_close(void* classifier, char** error_ms) {
  return mediapipe::tasks::c::vision::image_classifier::CppImageClassifierClose(
      classifier, error_ms);
}

}  // extern "C"
