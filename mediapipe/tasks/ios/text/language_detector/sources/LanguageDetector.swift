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
package import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_text_language_detector_language_detector_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
#endif

/// Identifies the language of input text using a MediaPipe TensorFlow Lite model.
///
/// - Note: This class is not thread-safe. Concurrent callers should serialize access (for example,
///   by isolating an instance inside a Swift `actor`).
public final class LanguageDetector {
  private let cDetector: MpLanguageDetectorPtr

  /// Initializes the underlying MediaPipe task graph using `options`.
  ///
  /// - Parameter options: Model asset and prediction-filtering settings.
  /// - Throws: `TasksError` if the model cannot be loaded or `options` are invalid.
  public init(options: LanguageDetectorOptions) throws {
    self.cDetector = try Self.createCDetector(options: options)
  }

  /// Initializes a detector from an on-disk `.tflite` model file using default filtering options.
  ///
  /// - Parameter modelPath: File-system path to the TensorFlow Lite model asset.
  /// - Throws: `TasksError` if `modelPath` is empty or the model cannot be loaded.
  public convenience init(modelPath: String) throws {
    var options = LanguageDetectorOptions()
    options.baseOptions.modelAssetPath = modelPath
    try self.init(options: options)
  }

  deinit {
    _ = MpLanguageDetectorClose(cDetector, nil)
  }

  /// Infers the most likely languages for `text`.
  ///
  /// - Parameter text: UTF-8 text to analyze.
  /// - Returns: Ranked language predictions matching the configured thresholds and filters.
  /// - Throws: `TasksError` if graph execution fails.
  public func detect(text: String) throws -> LanguageDetectorResult {
    var cResult = MpLanguageDetectorResult()
    try text.withCString { utf8Text in
      try MpStatus.check(defaultMessage: "Failed to detect language.") { errorMessage in
        MpLanguageDetectorDetect(cDetector, utf8Text, &cResult, &errorMessage)
      }
    }
    defer {
      MpLanguageDetectorCloseResult(&cResult)
    }
    return LanguageDetectorResult(cResult: cResult)
  }

  private static func createCDetector(
    options: LanguageDetectorOptions
  ) throws -> MpLanguageDetectorPtr {
    try options.withCOptions { cOptions in
      var mutableCOptions = cOptions
      var detectorPtr: MpLanguageDetectorPtr?
      try MpStatus.check(defaultMessage: "Failed to create LanguageDetector.") { errorMessage in
        MpLanguageDetectorCreate(&mutableCOptions, &detectorPtr, &errorMessage)
      }
      guard let detectorPtr else {
        throw TasksError.internalError("MpLanguageDetectorCreate returned a null pointer.")
      }
      return detectorPtr
    }
  }
}
