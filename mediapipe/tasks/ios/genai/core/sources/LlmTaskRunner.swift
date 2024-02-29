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
import MediaPipeTasksGenAIC

/// This class is used to create and call appropriate methods on the C `LlmInferenceEngine_Session`
/// to initialize, execute and terminate any MediaPipe `LlmInference` task.
public final class LlmTaskRunner {
  fileprivate typealias CLlmSession = UnsafeMutableRawPointer

  private let cLlmSession: CLlmSession

  /// Creates a new instance of `LlmTaskRunner` with the given session config.
  ///
  /// - Parameters:
  ///   - sessionConfig: C session config of type `LlmSessionConfig`.
  public init(sessionConfig: LlmSessionConfig) {
    /// No safe guards for session creation since the C APIs only throw fatal errors.
    /// `LlmInferenceEngine_CreateSession()` will always return a llm session if the call
    /// completes.
    self.cLlmSession = withUnsafePointer(to: sessionConfig) { LlmInferenceEngine_CreateSession($0) }
  }

  /// Invokes the C inference engine with the given input text to generate an array of `String`
  /// responses from the LLM.
  ///
  /// - Parameters:
  ///   - inputText: A `String` that is used to query the LLM.
  /// - Throws: An error if the LLM's response is invalid.
  public func predict(inputText: String) throws -> [String] {
    /// No safe guards for the call since the C++ APIs only throw fatal errors.
    /// `LlmInferenceEngine_Session_PredictSync()` will always return a `LlmResponseContext` if the
    /// call completes.
    var responseContext = inputText.withCString { cinputText in
      LlmInferenceEngine_Session_PredictSync(cLlmSession, cinputText)
    }

    defer {
      withUnsafeMutablePointer(to: &responseContext) {
        LlmInferenceEngine_CloseResponseContext($0)
      }
    }

    /// Throw an error if the response array is `NULL`.
    guard let cResponseArray = responseContext.response_array else {
      throw LlmInferenceError.invalidResponseError
    }

    var responseStrings: [String] = []

    for responseIndex in 0..<Int(responseContext.response_count) {
      guard let cResponseString = cResponseArray[responseIndex] else {
        throw LlmInferenceError.invalidResponseError
      }
      responseStrings.append(String(cString: cResponseString))
    }

    return responseStrings
  }

  deinit {
    LlmInferenceEngine_Session_Delete(cLlmSession)
  }

}
