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

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "mediapipe/tasks/c/components/containers/classification_result_converter.h"
#include "mediapipe/tasks/c/components/processors/classifier_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/cc/text/text_classifier/text_classifier.h"

using ::mediapipe::tasks::text::text_classifier::TextClassifier;

struct MpTextClassifierInternal {
  // The actual C++ classifier. Null if not yet initialized.
  std::unique_ptr<TextClassifier> instance{nullptr};
};

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

TextClassifier* GetCppClassifier(MpTextClassifierInternal* wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "TextClassifier is null.";
  return wrapper->instance.get();
}

}  // namespace

absl::Status CppTextClassifierCreate(const TextClassifierOptions& options,
                                     MpTextClassifierPtr* classifier) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::text::text_classifier::TextClassifierOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToClassifierOptions(options.classifier_options,
                                &cpp_options->classifier_options);

  auto cpp_classifier = TextClassifier::Create(std::move(cpp_options));
  if (!cpp_classifier.ok()) {
    return cpp_classifier.status();
  }
  *classifier =
      new MpTextClassifierInternal{.instance = std::move(*cpp_classifier)};
  return absl::OkStatus();
}

absl::Status CppTextClassifierClassify(MpTextClassifierPtr classifier,
                                       const char* utf8_str,
                                       TextClassifierResult* result) {
  auto* cpp_classifier = GetCppClassifier(classifier);
  auto cpp_result = cpp_classifier->Classify(utf8_str);
  if (!cpp_result.ok()) {
    return cpp_result.status();
  }
  CppConvertToClassificationResult(*cpp_result, result);
  return absl::OkStatus();
}

void CppTextClassifierCloseResult(TextClassifierResult* result) {
  CppCloseClassificationResult(result);
}

absl::Status CppTextClassifierClose(MpTextClassifierPtr classifier) {
  auto* cpp_classifier = GetCppClassifier(classifier);
  auto result = cpp_classifier->Close();
  if (!result.ok()) {
    return result;
  }
  delete classifier;
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::text::text_classifier

extern "C" {

MP_EXPORT MpStatus MpTextClassifierCreate(struct TextClassifierOptions* options,
                                          MpTextClassifierPtr* classifier,
                                          char** error_msg) {
  auto status =
      mediapipe::tasks::c::text::text_classifier::CppTextClassifierCreate(
          *options, classifier);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpTextClassifierClassify(MpTextClassifierPtr classifier,
                                            const char* utf8_str,
                                            TextClassifierResult* result,
                                            char** error_msg) {
  auto status =
      mediapipe::tasks::c::text::text_classifier::CppTextClassifierClassify(
          classifier, utf8_str, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpTextClassifierCloseResult(TextClassifierResult* result) {
  mediapipe::tasks::c::text::text_classifier::CppTextClassifierCloseResult(
      result);
}

MP_EXPORT MpStatus MpTextClassifierClose(MpTextClassifierPtr classifier,
                                         char** error_msg) {
  auto status =
      mediapipe::tasks::c::text::text_classifier::CppTextClassifierClose(
          classifier);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
