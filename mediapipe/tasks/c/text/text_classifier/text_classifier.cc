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
#include "mediapipe/tasks/c/components/processors/classifier_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/cc/text/text_classifier/text_classifier.h"

struct MpTextClassifierInternal {
  std::unique_ptr<::mediapipe::tasks::text::text_classifier::TextClassifier>
      classifier;
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
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::text::text_classifier::TextClassifier;

}  // namespace

MpStatus CppTextClassifierCreate(const TextClassifierOptions& options,
                                 MpTextClassifierPtr* classifier) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::text::text_classifier::TextClassifierOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToClassifierOptions(options.classifier_options,
                                &cpp_options->classifier_options);

  auto cpp_classifier = TextClassifier::Create(std::move(cpp_options));
  if (!cpp_classifier.ok()) {
    ABSL_LOG(ERROR) << "Failed to create TextClassifier: "
                    << cpp_classifier.status();
    return ToMpStatus(cpp_classifier.status());
  }
  *classifier =
      new MpTextClassifierInternal{.classifier = std::move(*cpp_classifier)};
  return kMpOk;
}

MpStatus CppTextClassifierClassify(MpTextClassifierPtr classifier,
                                   const char* utf8_str,
                                   TextClassifierResult* result) {
  auto cpp_classifier = classifier->classifier.get();
  auto cpp_result = cpp_classifier->Classify(utf8_str);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Classification failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }
  CppConvertToClassificationResult(*cpp_result, result);
  return kMpOk;
}

void CppTextClassifierCloseResult(TextClassifierResult* result) {
  CppCloseClassificationResult(result);
}

MpStatus CppTextClassifierClose(MpTextClassifierPtr classifier) {
  auto cpp_classifier = classifier->classifier.get();
  auto result = cpp_classifier->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close TextClassifier: " << result;
    return ToMpStatus(result);
  }
  delete classifier;
  return kMpOk;
}

}  // namespace mediapipe::tasks::c::text::text_classifier

extern "C" {

MP_EXPORT MpStatus MpTextClassifierCreate(struct TextClassifierOptions* options,
                                          MpTextClassifierPtr* classifier) {
  return mediapipe::tasks::c::text::text_classifier::CppTextClassifierCreate(
      *options, classifier);
}

MP_EXPORT MpStatus MpTextClassifierClassify(MpTextClassifierPtr classifier,
                                            const char* utf8_str,
                                            TextClassifierResult* result) {
  return mediapipe::tasks::c::text::text_classifier::CppTextClassifierClassify(
      classifier, utf8_str, result);
}

MP_EXPORT void MpTextClassifierCloseResult(TextClassifierResult* result) {
  mediapipe::tasks::c::text::text_classifier::CppTextClassifierCloseResult(
      result);
}

MP_EXPORT MpStatus MpTextClassifierClose(MpTextClassifierPtr classifier) {
  return mediapipe::tasks::c::text::text_classifier::CppTextClassifierClose(
      classifier);
}

}  // extern "C"
