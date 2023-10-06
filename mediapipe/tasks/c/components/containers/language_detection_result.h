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

#ifndef MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_LANGUAGE_DETECTION_RESULT_H_
#define MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_LANGUAGE_DETECTION_RESULT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A language code and its probability.
struct LanguageDetectorPrediction {
  // An i18n language / locale code, e.g. "en" for English, "uz" for Uzbek,
  // "ja"-Latn for Japanese (romaji).
  char* language_code;

  float probability;
};

// Task output.
struct LanguageDetectorResult {
  struct LanguageDetectorPrediction* predictions;

  // Keep the count of predictions.
  uint32_t predictions_count;
};

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_COMPONENTS_CONTAINERS_LANGUAGE_DETECTION_RESULT_H_
