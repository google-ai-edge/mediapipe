// Copyright 2026 The MediaPipe Authors.
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

import Foundation
package import third_party_mediapipe_tasks_c_text_language_detector_language_detector_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
#endif

/// A predicted language and its confidence score.
public struct LanguagePrediction: Sendable, Equatable, Hashable {
  /// BCP-47 / i18n language or locale identifier (for example, `"en"`, `"fr"`, or `"ja-Latn"`).
  public let languageCode: String

  /// Confidence score in `[0.0, 1.0]` assigned to `languageCode` by the model.
  public let probability: Float

  /// Creates a language prediction from a language identifier and confidence score.
  public init(languageCode: String, probability: Float) {
    self.languageCode = languageCode
    self.probability = probability
  }

  /// Creates a `LanguagePrediction` copied from a C `MpLanguageDetectorPrediction` struct.
  package init(cPrediction: MpLanguageDetectorPrediction) {
    self.languageCode = cPrediction.language_code.map { String(cString: $0) } ?? ""
    self.probability = cPrediction.probability
  }
}

/// Detection output produced by `LanguageDetector` for an input text.
public struct LanguageDetectorResult: Sendable, Equatable, Hashable {
  /// Ranked language candidates that passed filtering, ordered by descending probability.
  public let languages: [LanguagePrediction]

  /// Timestamp associated with the detection packet, in milliseconds.
  public let timestampInMilliseconds: Int

  /// Creates a detection result from ranked predictions and an optional packet timestamp.
  public init(languages: [LanguagePrediction], timestampInMilliseconds: Int = 0) {
    self.languages = languages
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  /// Creates a `LanguageDetectorResult` copied from a C `MpLanguageDetectorResult` struct.
  package init(cResult: MpLanguageDetectorResult, timestampInMilliseconds: Int = 0) {
    self.languages = Array(
      cArray: cResult.predictions,
      count: cResult.predictions_count,
      transform: LanguagePrediction.init(cPrediction:)
    )
    self.timestampInMilliseconds = timestampInMilliseconds
  }
}
