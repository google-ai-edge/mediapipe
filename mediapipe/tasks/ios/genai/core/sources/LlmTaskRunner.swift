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
  typealias CLlmSession = UnsafeMutableRawPointer

  private let cLlmSession: CLlmSession
  /// Creates a new instance of `LlmTaskRunner` with the given session config.
  ///
  /// - Parameters:
  ///   - sessionConfig: C session config of type `LlmSessionConfig`.
  public init(sessionConfig: LlmSessionConfig) {
    /// No safe guards for session creation since the C APIs only throw fatal errors.
    /// `LlmInferenceEngine_CreateSession()` will always return an llm session if the call
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
    var responseContext = inputText.withCString { cInputText in
      LlmInferenceEngine_Session_PredictSync(cLlmSession, cInputText)
    }

    defer {
      withUnsafeMutablePointer(to: &responseContext) {
        LlmInferenceEngine_CloseResponseContext($0)
      }
    }

    /// Throw an error if response is invalid.
    guard let responseStrings = LlmTaskRunner.responseStrings(from: responseContext) else {
      throw GenAiInferenceError.invalidResponse
    }

    return responseStrings
  }

  public func predict(
    inputText: String, progress: @escaping (_ partialResult: [String]?, _ error: Error?) -> Void,
    completion: @escaping (() -> Void)
  ) {

    /// `strdup(inputText)` prevents input text from being deallocated as long as callbacks are
    /// being invoked. `CallbackInfo` takes care of freeing the memory of `inputText` when it is
    /// deallocated.
    let callbackInfo = CallbackInfo(
      inputText: strdup(inputText), progress: progress, completion: completion)
    let callbackContext = UnsafeMutableRawPointer(Unmanaged.passRetained(callbackInfo).toOpaque())

    LlmInferenceEngine_Session_PredictAsync(cLlmSession, callbackContext, callbackInfo.inputText) {
      context, responseContext in
      guard let cContext = context else {
        return
      }

      /// `takeRetainedValue()` decrements the reference count incremented by `passRetained()`. Only
      /// take a retained value if the LLM has finished generating responses to prevent the context
      /// from being deallocated in between response generation.
      let cCallbackInfo =
        responseContext.done
        ? Unmanaged<CallbackInfo>.fromOpaque(cContext).takeRetainedValue()
        : Unmanaged<CallbackInfo>.fromOpaque(cContext).takeUnretainedValue()

      if let responseStrings = LlmTaskRunner.responseStrings(from: responseContext) {
        cCallbackInfo.progress(responseStrings, nil)
      } else {
        cCallbackInfo.progress(nil, GenAiInferenceError.invalidResponse)
      }

      /// Call completion callback if LLM has generated its last response.
      if responseContext.done {
        cCallbackInfo.completion()
      }
    }
  }

  deinit {
    LlmInferenceEngine_Session_Delete(cLlmSession)
  }
}

extension LlmTaskRunner {
  /// A wrapper class whose object will be used as the C++ callback context.
  /// The progress and completion callbacks cannot be invoked without a context.
  class CallbackInfo {
    typealias ProgressCallback = (_ partialResult: [String]?, _ error: Error?) -> Void
    typealias CompletionCallback = () -> Void

    let inputText: UnsafeMutablePointer<CChar>?
    let progress: ProgressCallback
    let completion: CompletionCallback

    init(
      inputText: UnsafeMutablePointer<CChar>?, progress: @escaping (ProgressCallback),
      completion: @escaping (CompletionCallback)
    ) {
      self.inputText = inputText
      self.progress = progress
      self.completion = completion
    }

    deinit {
      free(inputText)
    }
  }
}

extension LlmTaskRunner {
  private class func responseStrings(from responseContext: LlmResponseContext) -> [String]? {
    guard let cResponseArray = responseContext.response_array else {
      return nil
    }

    var responseStrings: [String] = []
    for responseIndex in 0..<Int(responseContext.response_count) {
      guard let cResponseString = cResponseArray[responseIndex] else {
        return nil
      }
      responseStrings.append(String(cString: cResponseString))
    }

    return responseStrings
  }
}
