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

#ifndef MEDIAPIPE_TASKS_C_TEXT_LANGUAGE_DETECTOR_LANGUAGE_DETECTOR_H_
#define MEDIAPIPE_TASKS_C_TEXT_LANGUAGE_DETECTOR_LANGUAGE_DETECTOR_H_

#include "mediapipe/tasks/c/components/processors/classifier_options.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/mp_status.h"

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MpLanguageDetectorInternal* MpLanguageDetectorPtr;

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

  // The count of predictions.
  uint32_t predictions_count;
};

// The options for configuring a MediaPipe language detector task.
struct LanguageDetectorOptions {
  // Base options for configuring MediaPipe Tasks, such as specifying the model
  // file with metadata, accelerator options, op resolver, etc.
  struct BaseOptions base_options;

  // Options for configuring the detector behavior, such as score threshold,
  // number of results, etc.
  struct ClassifierOptions classifier_options;
};

// Creates a LanguageDetector from the provided `options`.
// If successful, returns `kMpOk` and sets `*detector` to the new
// `MpLanguageDetectorPtr`.
MP_EXPORT MpStatus MpLanguageDetectorCreate(
    struct LanguageDetectorOptions* options, MpLanguageDetectorPtr* detector);

// Performs language detection on the input `utf8_str`.
// If successful, returns `kMpOk` and sets `*result` to the new
// `LanguageDetectorResult`.
MP_EXPORT MpStatus
MpLanguageDetectorDetect(MpLanguageDetectorPtr detector, const char* utf8_str,
                         struct LanguageDetectorResult* result);

// Frees the memory allocated inside a LanguageDetectorResult result. Does not
// free the result pointer itself.
MP_EXPORT void MpLanguageDetectorCloseResult(
    struct LanguageDetectorResult* result);

// Shuts down the LanguageDetector when all the work is done. Frees all memory.
// Returns `kMpOk` on success.
MP_EXPORT MpStatus MpLanguageDetectorClose(MpLanguageDetectorPtr detector);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_TEXT_LANGUAGE_DETECTOR_LANGUAGE_DETECTOR_H_
