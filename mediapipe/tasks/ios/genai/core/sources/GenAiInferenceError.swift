// Copyright 2024 The MediaPipe Authors.
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

/// Errors thrown by MediaPipe GenAI Tasks.
public enum GenAiInferenceError: Error {
  case invalidResponse
  case illegalMethodCall
  case failedToComputeSizeInTokens(String?)
  case failedToInitializeSession(String?)
  case failedToInitializeEngine(String?)
  case failedToAddQueryToSession(String, String?)
  case failedToCloneSession(String?)
}

extension GenAiInferenceError: LocalizedError {
  /// A localized description of the `GenAiInferenceError`.
  public var errorDescription: String? {
    switch self {
    case .invalidResponse:
      return "The response returned by the model is invalid."
    case .illegalMethodCall:
      return
        """
        Response generation is already in progress. The request in progress may have been \
        initated on the current session or on one of the sessions created from the `LlmInference` \
        that was used to create the current session.
        """
    case .failedToComputeSizeInTokens(let message):
      let explanation = message.flatMap { $0 } ?? "An internal error occurred."
      return "Failed to compute size of text in tokens: \(explanation)"
    case .failedToInitializeSession(let message):
      let explanation = message.flatMap { $0 } ?? "An internal error occurred."
      return "Failed to initialize LlmInference session: \(explanation)"
    case .failedToInitializeEngine(let message):
      let explanation = message.flatMap { $0 } ?? "An internal error occurred."
      return "Failed to initialize LlmInference engine: \(explanation)"
    case .failedToAddQueryToSession(let query, let message):
      let explanation = message.flatMap { $0 } ?? "An internal error occurred."
      return "Failed to add query: \(query) to LlmInference session: \(explanation)"
    case .failedToCloneSession(let message):
      let explanation = message.flatMap { $0 } ?? "An internal error occurred."
      return "Failed to clone LlmInference session: \(explanation)"
    }
  }
}

/// Protocol conformance for compatibility with `NSError`.
extension GenAiInferenceError: CustomNSError {
  static public var errorDomain: String {
    return "com.google.mediapipe.tasks.genai.inference"
  }

  public var errorCode: Int {
    switch self {
    case .invalidResponse:
      return 0
    case .illegalMethodCall:
      return 1
    case .failedToComputeSizeInTokens:
      return 2
    case .failedToInitializeSession:
      return 3
    case .failedToInitializeEngine:
      return 4
    case .failedToAddQueryToSession:
      return 5
    case .failedToCloneSession:
      return 6
    }
  }
}
