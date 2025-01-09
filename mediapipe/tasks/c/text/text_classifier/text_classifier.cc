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

#include "mediapipe/tasks/c/text/text_classifier/text_classifier.h"

#include <memory>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/tasks/c/components/containers/classification_result_converter.h"
#include "mediapipe/tasks/c/components/processors/classifier_options.h"
#include "mediapipe/tasks/c/components/processors/classifier_options_converter.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/cc/text/text_classifier/text_classifier.h"

namespace mediapipe::tasks::c::text::text_classifier {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseClassificationResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToClassificationResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToClassifierOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::text::text_classifier::TextClassifier;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
}

}  // namespace

TextClassifier* CppTextClassifierCreate(const TextClassifierOptions& options,
                                        char** error_msg) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::text::text_classifier::TextClassifierOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToClassifierOptions(options.classifier_options,
                                &cpp_options->classifier_options);

  auto classifier = TextClassifier::Create(std::move(cpp_options));
  if (!classifier.ok()) {
    ABSL_LOG(ERROR) << "Failed to create TextClassifier: "
                    << classifier.status();
    CppProcessError(classifier.status(), error_msg);
    return nullptr;
  }
  return classifier->release();
}

int CppTextClassifierClassify(void* classifier, const char* utf8_str,
                              TextClassifierResult* result, char** error_msg) {
  auto cpp_classifier = static_cast<TextClassifier*>(classifier);
  auto cpp_result = cpp_classifier->Classify(utf8_str);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Classification failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }
  CppConvertToClassificationResult(*cpp_result, result);
  return 0;
}

void CppTextClassifierCloseResult(TextClassifierResult* result) {
  CppCloseClassificationResult(result);
}

int CppTextClassifierClose(void* classifier, char** error_msg) {
  auto cpp_classifier = static_cast<TextClassifier*>(classifier);
  auto result = cpp_classifier->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close TextClassifier: " << result;
    return CppProcessError(result, error_msg);
  }
  delete cpp_classifier;
  return 0;
}

}  // namespace mediapipe::tasks::c::text::text_classifier

extern "C" {

void* text_classifier_create(struct TextClassifierOptions* options,
                             char** error_msg) {
  return mediapipe::tasks::c::text::text_classifier::CppTextClassifierCreate(
      *options, error_msg);
}

int text_classifier_classify(void* classifier, const char* utf8_str,
                             TextClassifierResult* result, char** error_msg) {
  return mediapipe::tasks::c::text::text_classifier::CppTextClassifierClassify(
      classifier, utf8_str, result, error_msg);
}

void text_classifier_close_result(TextClassifierResult* result) {
  mediapipe::tasks::c::text::text_classifier::CppTextClassifierCloseResult(
      result);
}

int text_classifier_close(void* classifier, char** error_ms) {
  return mediapipe::tasks::c::text::text_classifier::CppTextClassifierClose(
      classifier, error_ms);
}

}  // extern "C"
