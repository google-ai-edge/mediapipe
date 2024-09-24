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
/// to initialize, execute and terminate any MediaPipe `LlmInference.Session`.
final class LlmSessionRunner {
  typealias CLlmSession = UnsafeMutableRawPointer

  /// The underlying C LLM session managed by this `LlmSessionRunner`.
  private var cLlmSession: CLlmSession?

  /// Creates a new instance of `LlmSessionRunner` with the given C LLM session.
  ///
  /// - Parameters:
  ///   - cLlmSession: A session created by a C LLM engine.
  init(cLlmSession: UnsafeMutableRawPointer) {
    self.cLlmSession = cLlmSession
  }

  /// Adds query chunk to the C LLM session. This can be called multiple times to add multiple query
  /// chunks before calling `predict` or `predictAsync`. The query chunks will be processed in the
  /// order they are added, similar to a concatenated prompt, but able to be processed in chunks.
  ///
  /// - Parameters:
  ///   - inputText: Query chunk to be added to the C session.
  /// - Throws: An error if query chunk could not be added successfully.
  func addQueryChunk(inputText: String) throws {
    var cErrorMessage: UnsafeMutablePointer<CChar>? = nil

    guard
      (inputText.withCString { cInputText in
        LlmInferenceEngine_Session_AddQueryChunk(cLlmSession, cInputText, &cErrorMessage)
      }) == StatusCode.success.rawValue
    else {
      throw GenAiInferenceError.failedToAddQueryToSession(
        inputText, String(allocatedCErrorMessage: cErrorMessage))
    }
  }

  /// Invokes the C LLM session with the previously added query chunks synchronously to generate an
  /// array of `String` responses from the LLM.
  ///
  /// - Returns: Array of `String` responses from the LLM.
  /// - Throws: An error if the LLM's response is invalid.
  func predict() throws -> [String] {
    /// No safe guards for the call since the C++ APIs only throw fatal errors.
    /// `LlmInferenceEngine_Session_PredictSync()` will always return a `LlmResponseContext` if the
    /// call completes.
    var responseContext = LlmInferenceEngine_Session_PredictSync(cLlmSession)

    defer {
      withUnsafeMutablePointer(to: &responseContext) {
        LlmInferenceEngine_CloseResponseContext($0)
      }
    }

    /// Throw an error if response is invalid.
    guard let responseStrings = LlmSessionRunner.responseStrings(from: responseContext) else {
      throw GenAiInferenceError.invalidResponse
    }

    return responseStrings
  }

  /// Invokes the C LLM session with the previously added query chunks asynchronously to generate an
  /// array of `String` responses from the LLM. The `progress` callback returns the partial
  /// responses from the LLM or any errors. `completion` callback is invoked once the LLM is done
  /// generating responses.
  ///
  /// - Parameters:
  ///   - progress: A callback invoked when a partial response is available from the C LLM Session.
  ///   - completion: A callback invoked when the C LLM Session finishes response generation.
  /// - Throws: An error if the LLM's response is invalid.
  func predictAsync(
    progress: @escaping (_ partialResult: [String]?, _ error: Error?) -> Void,
    completion: @escaping (() -> Void)
  ) {
    let callbackInfo = CallbackInfo(progress: progress, completion: completion)
    let callbackContext = UnsafeMutableRawPointer(Unmanaged.passRetained(callbackInfo).toOpaque())

    LlmInferenceEngine_Session_PredictAsync(cLlmSession, callbackContext) {
      context, responseContext in
      guard let cContext = context else {
        return
      }

      guard let cResponse = responseContext?.pointee else {
        /// This failure is unlikely to happen. But throwing an error for the sake of completeness.
        ///
        /// If `responseContext` is nil, we have no way of knowing whether this was the last
        /// response. The code assumes that this was not the last response and lets the context
        /// continue in memory by taking an unretained value for it. This is to ensure that the
        /// context pointer returned by C in the subsequent callbacks is not dangling, thereby
        /// avoiding a seg fault. This has the downside that the context would continue indefinitely
        /// in memory if it was indeed the last response. The context would never get cleaned up.
        /// This will only be a problem if the failure happens on too many calls to `predictAsync`
        /// and leads to an out of memory error.
        let cCallbackInfo = Unmanaged<CallbackInfo>.fromOpaque(cContext).takeUnretainedValue()
        cCallbackInfo.progress(nil, GenAiInferenceError.invalidResponse)
        return
      }

      /// `takeRetainedValue()` decrements the reference count incremented by `passRetained()`. Only
      /// take a retained value if the LLM has finished generating responses to prevent the context
      /// from being deallocated in between response generation.
      let cCallbackInfo =
        cResponse.done
        ? Unmanaged<CallbackInfo>.fromOpaque(cContext).takeRetainedValue()
        : Unmanaged<CallbackInfo>.fromOpaque(cContext).takeUnretainedValue()

      if let responseStrings = LlmSessionRunner.responseStrings(from: cResponse) {
        cCallbackInfo.progress(responseStrings, nil)
      } else {
        cCallbackInfo.progress(nil, GenAiInferenceError.invalidResponse)
      }

      LlmInferenceEngine_CloseResponseContext(responseContext)

      /// Call completion callback if LLM has generated its last response.
      if cResponse.done {
        cCallbackInfo.completion()
      }
    }
  }

  /// Invokes the C LLM session to tokenize an input prompt using a pre-existing processor and
  /// returns its length in tokens.
  ///
  /// - Parameters:
  ///   - text: An input prompt.
  /// - Returns: Length of the input prompt in tokens.
  /// - Throws: An error if the number of tokens in the input prompt cannot be calculated.
  func sizeInTokens(text: String) throws -> Int {
    var cErrorMessage: UnsafeMutablePointer<CChar>?

    let sizeInTokens = text.withCString { cText in
      LlmInferenceEngine_Session_SizeInTokens(cLlmSession, cText, &cErrorMessage)
    }

    guard sizeInTokens > -1 else {
      throw GenAiInferenceError.failedToComputeSizeInTokens(
        String(allocatedCErrorMessage: cErrorMessage))
    }

    return Int(sizeInTokens)
  }

  /// Creates a clone of the current instance of `LlmSessionRunner` by cloning the underlying C
  /// LLM session.
  /// Note: Currently, this method is only available for GPU models.
  ///
  /// - Returns: Cloned `LlmSessionRunner`.
  /// - Throws: An error if the underlying C LLM session could not be cloned.
  func clone() throws -> LlmSessionRunner {
    var clonedCLlmSession: UnsafeMutableRawPointer?
    var cErrorMessage: UnsafeMutablePointer<CChar>? = nil
    guard
      LlmInferenceEngine_Session_Clone(cLlmSession, &clonedCLlmSession, &cErrorMessage)
        == StatusCode.success.rawValue,
      let clonedCLlmSession
    else {
      throw GenAiInferenceError.failedToCloneSession(String(allocatedCErrorMessage: cErrorMessage))
    }

    return LlmSessionRunner(cLlmSession: clonedCLlmSession)
  }

  deinit {
    LlmInferenceEngine_Session_Delete(cLlmSession)
  }
}

extension LlmSessionRunner {
  /// A wrapper class whose object will be used as the C++ callback context.
  /// The progress and completion callbacks cannot be invoked without a context.
  class CallbackInfo {
    typealias ProgressCallback = (_ partialResult: [String]?, _ error: Error?) -> Void
    typealias CompletionCallback = () -> Void

    let progress: ProgressCallback
    let completion: CompletionCallback

    init(
      progress: @escaping (ProgressCallback),
      completion: @escaping (CompletionCallback)
    ) {
      self.progress = progress
      self.completion = completion
    }
  }
}

extension LlmSessionRunner {
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
