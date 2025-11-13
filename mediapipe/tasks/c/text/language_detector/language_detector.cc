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
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/text/language_detector/language_detector_result_converter.h"
#include "mediapipe/tasks/cc/text/language_detector/language_detector.h"

struct MpLanguageDetectorInternal {
  std::unique_ptr<::mediapipe::tasks::text::language_detector::LanguageDetector>
      detector;
};

namespace mediapipe::tasks::c::text::language_detector {

namespace {

using ::mediapipe::tasks::c::components::containers::
    CppCloseLanguageDetectorResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToLanguageDetectorResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToClassifierOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::text::language_detector::LanguageDetector;

}  // namespace

MpStatus CppLanguageDetectorCreate(const LanguageDetectorOptions& options,
                                   MpLanguageDetectorPtr* detector) {
  auto cpp_options = std::make_unique<
      ::mediapipe::tasks::text::language_detector::LanguageDetectorOptions>();

  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToClassifierOptions(options.classifier_options,
                                &cpp_options->classifier_options);

  auto cpp_detector = LanguageDetector::Create(std::move(cpp_options));
  if (!cpp_detector.ok()) {
    ABSL_LOG(ERROR) << "Failed to create LanguageDetector: "
                    << cpp_detector.status();
    return ToMpStatus(cpp_detector.status());
  }
  *detector =
      new MpLanguageDetectorInternal{.detector = std::move(*cpp_detector)};
  return kMpOk;
}

MpStatus CppLanguageDetectorDetect(MpLanguageDetectorPtr detector,
                                   const char* utf8_str,
                                   LanguageDetectorResult* result) {
  auto cpp_detector = detector->detector.get();
  auto cpp_result = cpp_detector->Detect(utf8_str);
  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Language Detector failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  }

  CppConvertToLanguageDetectorResult(*cpp_result, result);
  return kMpOk;
}

void CppLanguageDetectorCloseResult(LanguageDetectorResult* result) {
  CppCloseLanguageDetectorResult(result);
}

MpStatus CppLanguageDetectorClose(MpLanguageDetectorPtr detector) {
  auto cpp_detector = detector->detector.get();
  auto result = cpp_detector->Close();
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "Failed to close LanguageDetector: " << result;
    return ToMpStatus(result);
  }
  delete detector;
  return kMpOk;
}

}  // namespace mediapipe::tasks::c::text::language_detector

extern "C" {

MP_EXPORT MpStatus MpLanguageDetectorCreate(
    struct LanguageDetectorOptions* options, MpLanguageDetectorPtr* detector) {
  return mediapipe::tasks::c::text::language_detector::
      CppLanguageDetectorCreate(*options, detector);
}

MP_EXPORT MpStatus MpLanguageDetectorDetect(MpLanguageDetectorPtr detector,
                                            const char* utf8_str,
                                            LanguageDetectorResult* result) {
  return mediapipe::tasks::c::text::language_detector::
      CppLanguageDetectorDetect(detector, utf8_str, result);
}

MP_EXPORT void MpLanguageDetectorCloseResult(LanguageDetectorResult* result) {
  mediapipe::tasks::c::text::language_detector::CppLanguageDetectorCloseResult(
      result);
}

MP_EXPORT MpStatus MpLanguageDetectorClose(MpLanguageDetectorPtr detector) {
  return mediapipe::tasks::c::text::language_detector::CppLanguageDetectorClose(
      detector);
}

}  // extern "C"
