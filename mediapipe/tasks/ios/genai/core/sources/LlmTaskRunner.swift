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

/// This class is used to create and call appropriate methods on the C `LlmInferenceEngine` to
/// initialize an LLMInference task and create prompt sessions.
final class LlmTaskRunner {
  typealias CLlmEngine = UnsafeMutableRawPointer

  /// The underlying C LLM engine created and managed by this `LlmTaskRunner`.
  private var cLlmEngine: CLlmEngine?

  /// Creates a new instance of `LlmTaskRunner` with the given model settings.
  ///
  /// - Parameters:
  ///   - modelSettings: C model settings of type `LlmModelSettings`.
  /// - Throws: An error if the engine could not be initialized.
  init(modelSettings: LlmModelSettings) throws {
    var cErrorMessage: UnsafeMutablePointer<CChar>? = nil
    guard
      (withUnsafePointer(to: modelSettings) {
        LlmInferenceEngine_CreateEngine($0, &self.cLlmEngine, &cErrorMessage)
      }) == StatusCode.success.rawValue
    else {
      throw GenAiInferenceError.failedToInitializeEngine(
        String(allocatedCErrorMessage: cErrorMessage))
    }
  }

  /// Creates a new C LLM session from the current C engine and returns an `LlmSessionRunner`
  /// that wraps around the newly created C session. The session runner is responsible for managing
  /// its underlying C session.
  ///
  /// Note: On each invocation, this method returns a new instance of the session runner configured
  /// to the values provided in the session config. Thus, if you provide the session config of a
  /// currently active LLM session, this method will create and return a duplicate session runner
  /// configured to the same values. The task runner does not keep track of the currently active
  /// session runners.
  ///
  /// - Parameters:
  ///   - sessionConfig: C session config of type `LlmSessionConfig` that configures how to execute
  /// the model.
  /// - Returns: A new instance of `LlmSessionRunner`.
  /// - Throws: An error if the engine could not be initialized.
  func createSessionRunner(sessionConfig: LlmSessionConfig) throws -> LlmSessionRunner {
    var cErrorMessage: UnsafeMutablePointer<CChar>?
    var cLlmSession: UnsafeMutableRawPointer?

    guard
      (withUnsafePointer(to: sessionConfig) {
        LlmInferenceEngine_CreateSession(cLlmEngine, $0, &cLlmSession, &cErrorMessage)
      }) == StatusCode.success.rawValue,
      let cLlmSession
    else {
      throw GenAiInferenceError.failedToInitializeSession(
        String(allocatedCErrorMessage: cErrorMessage))
    }

    let llmSessionRunner = LlmSessionRunner(cLlmSession: cLlmSession)
    return llmSessionRunner
  }

  deinit {
    LlmInferenceEngine_Engine_Delete(cLlmEngine)
  }
}

extension String {
  init?(allocatedCErrorMessage: UnsafeMutablePointer<CChar>?) {
    guard let allocatedCErrorMessage else {
      return nil
    }

    self.init(cString: allocatedCErrorMessage)
    free(allocatedCErrorMessage)
  }
}

enum StatusCode: Int {
  case success = 0
}
