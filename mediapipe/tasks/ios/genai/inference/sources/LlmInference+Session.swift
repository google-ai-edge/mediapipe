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

extension LlmInference {

  /// An `LlmInference` Session that can be used to execute queries using the LLM that was used to
  /// initialize the `LlmInference`task.
  /// You can create multiple query sessions using the same `LlmInference` which can be active at
  /// the same time.
  /// You can also clone an existing session and continue querying the LLM from where you left off.
  ///
  /// Note: Inherits from `NSObject` for Objective C interoperability.
  @objc(MPPLLMInferenceSession) public final class Session: NSObject {
    private static let responseGenerationInProgressQueueName =
      "com.google.mediapipe.genai.isResponseGenerationInProgressQueue"

    // Session runner that manages the creation, deletion and execution of the underlying C session.
    private let llmSessionRunner: LlmSessionRunner

    // Queue that restricts access to the response generation functions simultaneously.
    private let responseGenerationInProgressQueue = DispatchQueue(
      label: Session.responseGenerationInProgressQueueName,
      attributes: .concurrent)

    /// Tracks whether a response generation is in progress.
    /// Readers writers lock to prevent race condition as this variable can be accessed from multiple
    /// threads.
    private var responseGenerationInProgressInternal = false
    private var responseGenerationInProgress: Bool {
      get {
        responseGenerationInProgressQueue.sync {
          return self.responseGenerationInProgressInternal
        }
      }
      set {
        responseGenerationInProgressQueue.async(flags: .barrier) {
          self.responseGenerationInProgressInternal = newValue
        }
      }
    }

    /// Creates a new instance of `LlmInference` with the given options. 
    /// Ensure that the instance of `LlmInference` passed to this function must be alive through 
    /// the lifetime of the current session. Using a session beyond the life time of the 
    /// `LlmInference` used to create it will lead to undefined behaviour.
    ///
    /// - Parameters:
    ///   - options: The options of type `LlmInference.Options` to use for configuring the
    /// `LlmInference`.
    /// - Throws: An error if `LlmInference` instance could not be initialized.
    @objc public init(llmInference: LlmInference, options: Options) throws {
      var sessionConfig = LlmSessionConfig(
        topk: options.topk,
        topp: options.topp,
        temperature: options.temperature,
        random_seed: options.randomSeed,
        lora_path: nil)

      /// If `loraPath` is != nil, modify session config with the corresponding C string and invoke
      /// the method to create session runner within the scope where the C String of the `loraPath`
      /// is alive. Otherwise, create the session runner with the previously created config.
      /// C++ copies `loraPath`, hence it need not be retained by this class.
      llmSessionRunner =
        try options.loraPath?.withCString { loraPath in
          sessionConfig.lora_path = loraPath
          return try llmInference.createSessionRunner(sessionConfig: sessionConfig)
        } ?? llmInference.createSessionRunner(sessionConfig: sessionConfig)

      super.init()
    }

    /// A convenience initializer that creates a new instance of `LlmInference.Session` from an
    /// the given `llmInference` and default options.
    /// Ensure that the instance of `LlmInference` passed to this function must be alive through 
    /// the lifetime of the current session. Using a session beyond the life time of the 
    /// `LlmInference` used to create it will lead to undefined behaviour.
    ///
    /// - Parameters:
    ///   - llmInference: An instance of `LlmInference` from which the session must be created.
    /// - Throws: An error if a new session could not be created from the give `llmInference`.
    @objc public convenience init(llmInference: LlmInference) throws {
      let options = Options()
      try self.init(llmInference: llmInference, options: options)
    }

    /// Creates a new instance of `LlmInference.Session` with the given session runner.
    /// This initializer is used by `clone()` to create a new `LlmInference.Session` using a
    /// cloned session runner.
    /// - Parameters:
    ///   - llmSessionRunner: An instance of `LlmSessionRunner` using which the session must be
    /// created.
    init(llmSessionRunner: LlmSessionRunner) {
      self.llmSessionRunner = llmSessionRunner
      super.init()
    }

    /// Adds a query chunk to the session. This method can be called multiple times to add multiple
    /// query chunks before calling `generateResponse()` or `generateResponseAsync()`. The query
    /// chunks will be processed in the order they are added, similar to a concatenated prompt,
    /// but able to be processed in chunks.
    @objc public func addQueryChunk(inputText: String) throws {
      try llmSessionRunner.addQueryChunk(inputText: inputText)
    }

    /// Generates a response based on the previously added query chunks synchronously. Use
    /// `addQueryChunk(inputText:)` to add atleast one query chunk before calling this function.
    ///
    /// - Throws: An error if the LLM's response is invalid.
    @objc public func generateResponse() throws -> String {

      /// Disallow response generation if another response generation call is already in progress.
      try shouldContinueWithResponseGeneration()

      let tokens = try llmSessionRunner.predict()

      responseGenerationInProgress = false

      guard let humanReadableLlmResponse = Session.humanReadableString(llmResponses: tokens)
      else {
        throw GenAiInferenceError.invalidResponse
      }

      return humanReadableLlmResponse
    }

    /// Generates a response based on the previously added query chunks asynchronously. The
    /// `progress` callback returns the partial responses from the LLM or any errors.
    /// `completion` callback is invoked once the LLM is done generating responses.
    /// Use `addQueryChunk(inputText:)` to add atleast one query chunk before calling this function.
    ///
    /// - Parameters:
    ///   - progress: A callback invoked when a partial response is available from the LLM.
    ///   - completion: A callback invoked when the LLM finishes response generation.
    /// - Throws: An error if the LLM's response is invalid.
    @objc public func generateResponseAsync(
      progress: @escaping (_ partialResponse: String?, _ error: Error?) -> Void,
      completion: @escaping (() -> Void)
    ) throws {
      /// Disallow response generation if another response generation call is already in progress.
      try shouldContinueWithResponseGeneration()

      /// Used to make a decision about whitespace stripping.
      var receivedFirstToken = true

      try llmSessionRunner.predictAsync(
        progress: { partialResponseStrings, error in
          guard let responseStrings = partialResponseStrings,
            let humanReadableLlmResponse = Session.humanReadableString(
              llmResponses: responseStrings, stripLeadingWhitespaces: receivedFirstToken)
          else {
            progress(nil, GenAiInferenceError.invalidResponse)
            return
          }

          /// Reset state after first response is processed.
          receivedFirstToken = false

          progress(humanReadableLlmResponse, nil)
        },
        completion: { [weak self] in
          self?.responseGenerationInProgress = false
          completion()
        })
    }

    /// Generates a response based on the previously added query chunks asynchronously.
    /// Use `addQueryChunk(inputText:)` to add atleast one query chunk before calling this function.
    ///
    /// - Returns: An async throwing stream that contains the partial responses from the LLM.
    @available(iOS 13, macOS 10.15, tvOS 13, watchOS 6, *)
    public func generateResponseAsync() -> AsyncThrowingStream<String, Error> {
      AsyncThrowingStream { continuation in
        do {
          try generateResponseAsync(
            progress: { partialResponse, error in
              if let error {
                continuation.finish(throwing: error)
              } else if let partialResponse {
                continuation.yield(partialResponse)
              }
            },
            completion: {
              continuation.finish()
            })
        } catch {
          continuation.finish(throwing: error)
        }
      }
    }

    /// Returns the size in tokens of the provided text.
    /// You may use this function to verify this size before submitting the prompt to ensure it
    /// doesn't exceed the configured maximum token size.
    ///
    /// - Parameters:
    ///   - text: The input text whose size in tokens is to be counted.
    /// - Returns: The size in tokens of the provided text.
    /// - Throws: An error if calculating the size in tokens of the provided text fails.
    public func sizeInTokens(text: String) throws -> Int {
      return try llmSessionRunner.sizeInTokens(text: text)
    }

    /// Clones the current session.
    ///
    /// - Returns: A new instance of `Session` which is cloned from the current session.
    /// - Throws: An error if cloning the current session fails.
    public func clone() throws -> Session {
      let clonedSessionRunner = try llmSessionRunner.clone()
      return Session(llmSessionRunner: clonedSessionRunner)
    }

    /// Throw error if response generation is in progress or update response generation state.
    private func shouldContinueWithResponseGeneration() throws {
      if responseGenerationInProgress {
        throw GenAiInferenceError.illegalMethodCall
      }

      responseGenerationInProgress = true
    }

    private static func humanReadableString(
      llmResponses: [String], stripLeadingWhitespaces: Bool = true
    ) -> String? {
      guard let llmResponse = llmResponses.first else {
        return nil
      }
      return llmResponse.humanReadableString(stripLeadingWhitespaces: stripLeadingWhitespaces)
    }

  }
}

// Extension to `LlmInference.Session` for defining `LlmInference.Session.Options`
extension LlmInference.Session {
  /// Options for setting up a `LlmInference.Session`.
  ///
  /// Note: Inherits from `NSObject` for Objective C interoperability.
  @objc(MPPLLMInferenceSessionOptions) public final class Options: NSObject {
    /// The top K number of tokens to be sampled from for each decoding step. A value of 1 means
    /// greedy decoding. Defaults to 40.
    @objc public var topk: Int = 40

    /// Maximum cumulative probability over the tokens to sample from in each decoding step for
    /// top-p / nucleus sampling.
    @objc public var topp: Float = 1.0

    /// The randomness when decoding the next token. A value of 0.0f means greedy decoding. Defaults
    /// to 0.8.
    @objc public var temperature: Float = 0.8

    /// The random seed for sampling tokens.
    @objc public var randomSeed: Int = 0

    /// The absolute path to the LoRA model asset bundle stored locally on the device. Optional.
    /// This is only compatible with GPU models.
    @objc public var loraPath: String?
  }
}

/// An extension to `String` to add some utility functions.
extension String {
  private static let tokenSplitter = "▁"
  /// Note this is NOT an underscore: ▁(U+2581)
  private static let newLine = "<0x0A>"
  private static let eod = "\\[eod\\]"

  fileprivate func humanReadableString(stripLeadingWhitespaces: Bool = true) -> String? {
    var humanReadableString = self.replacingOccurrences(of: String.tokenSplitter, with: " ")
      .replacingOccurrences(of: String.newLine, with: "\n")
    humanReadableString =
      stripLeadingWhitespaces
      ? humanReadableString.trimmingCharacters(in: .whitespaces) : humanReadableString
    return humanReadableString.components(separatedBy: String.eod).first
  }
}
