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

#include "mediapipe/tasks/c/components/containers/classification_result.h"
#include "mediapipe/tasks/c/components/processors/classifier_options.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/cc/text/text_classifier/text_classifier.h"

namespace mediapipe::tasks::c::text::text_classifier {

namespace {

using ::mediapipe::tasks::c::components::containers::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToClassificationResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToClassifierOptions;
using ::mediapipe::tasks::text::text_classifier::TextClassifier;
}  // namespace

TextClassifier* CppTextClassifierCreate(TextClassifierOptions options) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::text::text_classifier::TextClassifierOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToClassifierOptions(options.classifier_options,
                                &cpp_options->classifier_options);

  auto classifier = TextClassifier::Create(std::move(cpp_options));
  if (!classifier.ok()) {
    LOG(ERROR) << "Failed to create TextClassifier: " << classifier.status();
    return nullptr;
  }
  return classifier->release();
}

bool CppTextClassifierClassify(void* classifier, char* utf8_str,
                               TextClassifierResult* result) {
  auto cpp_classifier = static_cast<TextClassifier*>(classifier);
  auto cpp_result = cpp_classifier->Classify(utf8_str);
  if (!cpp_result.ok()) {
    LOG(ERROR) << "Classification failed: " << cpp_result.status();
    return false;
  }
  CppConvertToClassificationResult(*cpp_result, result);
  return true;
}

void CppTextClassifierClose(void* classifier) {
  auto cpp_classifier = static_cast<TextClassifier*>(classifier);
  auto result = cpp_classifier->Close();
  if (!result.ok()) {
    LOG(ERROR) << "Failed to close TextClassifier: " << result;
  }
  delete cpp_classifier;
}

}  // namespace mediapipe::tasks::c::text::text_classifier

extern "C" {

void* text_classifier_create(struct TextClassifierOptions options) {
  return mediapipe::tasks::c::text::text_classifier::CppTextClassifierCreate(
      options);
}

bool text_classifier_classify(void* classifier, char* utf8_str,
                              TextClassifierResult* result) {
  return mediapipe::tasks::c::text::text_classifier::CppTextClassifierClassify(
      classifier, utf8_str, result);
}

void text_classifier_close(void* classifier) {
  mediapipe::tasks::c::text::text_classifier::CppTextClassifierClose(
      classifier);
}

}  // extern "C"
