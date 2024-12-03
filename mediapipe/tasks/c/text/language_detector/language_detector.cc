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

#include "mediapipe/tasks/c/text/language_detector/language_detector.h"

#include <memory>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/tasks/c/components/processors/classifier_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/text/language_detector/language_detector_result_converter.h"
#include "mediapipe/tasks/cc/text/language_detector/language_detector.h"

namespace mediapipe::tasks::c::text::language_detector {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseLanguageDetectorResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToLanguageDetectorResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToClassifierOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::text::language_detector::LanguageDetector;

int CppProcessError(absl::Status status, char** error_msg) {
  if (error_msg) {
    *error_msg = strdup(status.ToString().c_str());
  }
  return status.raw_code();
}

}  // namespace

LanguageDetector* CppLanguageDetectorCreate(
    const LanguageDetectorOptions& options, char** error_msg) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::text::language_detector::LanguageDetectorOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToClassifierOptions(options.classifier_options,
                                &cpp_options->classifier_options);

  auto detector = LanguageDetector::Create(std::move(cpp_options));
  if (!detector.ok()) {
    ABSL_LOG(ERROR) << "Failed to create LanguageDetector: "
                    << detector.status();
    CppProcessError(detector.status(), error_msg);
    return nullptr;
  }
  return detector->release();
}

int CppLanguageDetectorDetect(void* detector, const char* utf8_str,
                              LanguageDetectorResult* result,
                              char** error_msg) {
  auto cpp_detector = static_cast<LanguageDetector*>(detector);
  auto cpp_result = cpp_detector->Detect(utf8_str);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Language Detector failed: " << cpp_result.status();
    return CppProcessError(cpp_result.status(), error_msg);
  }

  CppConvertToLanguageDetectorResult(*cpp_result, result);
  return 0;
}

void CppLanguageDetectorCloseResult(LanguageDetectorResult* result) {
  CppCloseLanguageDetectorResult(result);
}

int CppLanguageDetectorClose(void* detector, char** error_msg) {
  auto cpp_detector = static_cast<LanguageDetector*>(detector);
  auto result = cpp_detector->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close LanguageDetector: " << result;
    return CppProcessError(result, error_msg);
  }
  delete cpp_detector;
  return 0;
}

}  // namespace mediapipe::tasks::c::text::language_detector

extern "C" {

void* language_detector_create(struct LanguageDetectorOptions* options,
                               char** error_msg) {
  return mediapipe::tasks::c::text::language_detector::
      CppLanguageDetectorCreate(*options, error_msg);
}

int language_detector_detect(void* detector, const char* utf8_str,
                             LanguageDetectorResult* result, char** error_msg) {
  return mediapipe::tasks::c::text::language_detector::
      CppLanguageDetectorDetect(detector, utf8_str, result, error_msg);
}

void language_detector_close_result(LanguageDetectorResult* result) {
  mediapipe::tasks::c::text::language_detector::CppLanguageDetectorCloseResult(
      result);
}

int language_detector_close(void* detector, char** error_ms) {
  return mediapipe::tasks::c::text::language_detector::CppLanguageDetectorClose(
      detector, error_ms);
}

}  // extern "C"
