// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.tasks.text.languagedetector;

import com.google.auto.value.AutoValue;

/** A language code and its probability. Used as part of the output of {@link LanguageDetector}. */
@AutoValue
public abstract class LanguagePrediction {
  /**
   * Creates a {@link LanguageDetectorPrediction} instance.
   *
   * @param languageCode An i18n language / locale code, e.g. "en" for English, "uz" for Uzbek,
   *     "ja"-Latn for Japanese (romaji).
   * @param probability The probability for the prediction.
   */
  public static LanguagePrediction create(String languageCode, float probability) {
    return new AutoValue_LanguagePrediction(languageCode, probability);
  }

  /** The i18n language / locale code for the prediction. */
  public abstract String languageCode();

  /** The probability for the prediction. */
  public abstract float probability();
}
