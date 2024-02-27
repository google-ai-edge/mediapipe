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

/// Errors thrown by MediaPipe `LlmInference`.
public enum LlmInferenceError: Error {
  case invalidResponseError
}

extension LlmInferenceError: LocalizedError {
  /// A localized description of the `LlmInferenceError`.
  public var errorDescription: String? {
    switch self {
    case .invalidResponseError:
      return "The response returned by the large language model is invalid."
    }
  }
}

/// Protocol conformance for compatibilty with `NSError`.
extension LlmInferenceError: CustomNSError {
  static public var errorDomain: String {
    return "com.google.mediapipe.tasks.genai.inference"
  }

  public var errorCode: Int {
    switch self {
    case .invalidResponseError:
      return 0
    }
  }
}
