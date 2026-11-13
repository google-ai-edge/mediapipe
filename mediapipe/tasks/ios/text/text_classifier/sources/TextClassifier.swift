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
package import third_party_mediapipe_tasks_c_components_containers_classification_result
package import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_text_text_classifier_text_classifier_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
#endif

/// Performs classification on text using a TensorFlow Lite model with optional metadata.
///
/// Supports BERT-tokenized, regex-tokenized, and raw string-input classification models, returning
/// per-head category predictions filtered by the configured `TextClassifierOptions`.
///
/// - Note: This class is not thread-safe. Concurrent callers should serialize access (for example,
///   by isolating an instance inside a Swift `actor`).
public final class TextClassifier {
  private let cClassifier: MpTextClassifierPtr

  /// Creates a text classifier and initializes the underlying MediaPipe task graph.
  ///
  /// - Parameter options: Configuration specifying the model asset and classification filters.
  /// - Throws: `TasksError` if the model asset is missing or graph initialization fails.
  public init(options: TextClassifierOptions) throws {
    let hasModelPath = !options.baseOptions.modelAssetPath.isEmpty
    let hasModelBuffer = options.baseOptions.modelAssetBuffer.map { !$0.isEmpty } ?? false
    guard hasModelPath || hasModelBuffer else {
      throw TasksError.invalidArgument(
        "TextClassifierOptions.baseOptions must specify a non-empty modelAssetPath or "
          + "modelAssetBuffer."
      )
    }

    self.cClassifier = try options.withCOptions { cOptions in
      var mutableOptions = cOptions
      var classifierPtr: MpTextClassifierPtr?
      try MpStatus.check(defaultMessage: "Failed to create TextClassifier.") { errorMessage in
        MpTextClassifierCreate(&mutableOptions, &classifierPtr, &errorMessage)
      }
      guard let classifierPtr else {
        throw TasksError.internalError("MpTextClassifierCreate returned a null pointer.")
      }
      return classifierPtr
    }
  }

  /// Creates a text classifier from a local model file path using default classifier options.
  ///
  /// - Parameter modelPath: File-system path to a TensorFlow Lite model asset.
  /// - Throws: `TasksError` if the model file cannot be loaded or initialized.
  public convenience init(modelPath: String) throws {
    let baseOptions = BaseOptions(modelAssetPath: modelPath)
    try self.init(options: TextClassifierOptions(baseOptions: baseOptions))
  }

  deinit {
    _ = MpTextClassifierClose(cClassifier, nil)
  }

  /// Classifies the input string and returns predicted categories per model output head.
  ///
  /// - Parameter text: UTF-8 string to classify.
  /// - Returns: Classification predictions across all output heads of the model.
  /// - Throws: `TasksError` if inference fails.
  public func classify(text: String) throws -> TextClassifierResult {
    var cResult = MpTextClassifierResult()
    defer {
      MpTextClassifierCloseResult(&cResult)
    }
    try text.withCString { utf8Text in
      try MpStatus.check(defaultMessage: "Failed to classify text.") { errorMessage in
        MpTextClassifierClassify(cClassifier, utf8Text, &cResult, &errorMessage)
      }
    }
    return TextClassifierResult(cResult: cResult)
  }
}
