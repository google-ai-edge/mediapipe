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

#include "mediapipe/tasks/c/text/language_detector/language_detector_result_converter.h"

#include <cstdint>
#include <cstdlib>

#include "mediapipe/tasks/c/text/language_detector/language_detector.h"
#include "mediapipe/tasks/cc/text/language_detector/language_detector.h"

namespace mediapipe::tasks::c::components::containers {

void CppConvertToLanguageDetectorResult(
    const mediapipe::tasks::text::language_detector::LanguageDetectorResult& in,
    LanguageDetectorResult* out) {
  out->predictions_count = in.size();
  out->predictions =
      out->predictions_count
          ? new LanguageDetectorPrediction[out->predictions_count]
          : nullptr;

  for (uint32_t i = 0; i < out->predictions_count; ++i) {
    auto language_detection_prediction_in = in[i];
    auto& language_detection_prediction_out = out->predictions[i];
    language_detection_prediction_out.probability =
        language_detection_prediction_in.probability;
    language_detection_prediction_out.language_code =
        strdup(language_detection_prediction_in.language_code.c_str());
  }
}

void CppCloseLanguageDetectorResult(LanguageDetectorResult* in) {
  for (uint32_t i = 0; i < in->predictions_count; ++i) {
    auto prediction_in = in->predictions[i];

    free(prediction_in.language_code);
    prediction_in.language_code = nullptr;
  }
  delete[] in->predictions;
  in->predictions = nullptr;
}

}  // namespace mediapipe::tasks::c::components::containers
