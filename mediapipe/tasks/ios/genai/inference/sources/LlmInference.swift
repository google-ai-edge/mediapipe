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

/// A MediaPipe task that performs inference using a given Large Language Model.
///
/// Note: Inherits from `NSObject` for Objective C interoperability.
@objc(MPPLLMInference) public final class LlmInference: NSObject {
  private static let numberOfDecodeStepsPerSync = 3
  private static let sequenceBatchSize = 0
  private static let responseGenerationInProgressQueueName =
    "com.google.mediapipe.genai.isResponseGenerationInProgressQueue"

  private let llmTaskRunner: LlmTaskRunner

  private let responseGenerationInProgressQueue = DispatchQueue(
    label: LlmInference.responseGenerationInProgressQueueName,
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
  ///
  /// - Parameters:
  ///   - options: The options of type `LlmInference.Options` to use for configuring the
  /// `LlmInference`.
  @objc public init(options: Options) {
    let modelPath = strdup(options.modelPath)
    let cacheDirectory = strdup(FileManager.default.temporaryDirectory.path)

    defer {
      free(modelPath)
      free(cacheDirectory)
    }

    let sessionConfig = LlmSessionConfig(
      model_path: modelPath,
      cache_dir: cacheDirectory,
      sequence_batch_size: LlmInference.sequenceBatchSize,
      num_decode_steps_per_sync: LlmInference.numberOfDecodeStepsPerSync,
      max_tokens: options.maxTokens,
      topk: options.topk,
      temperature: options.temperature,
      random_seed: options.randomSeed)
    llmTaskRunner = LlmTaskRunner(sessionConfig: sessionConfig)

    super.init()
  }

  /// A convenience initializer that creates a new instance of `LlmInference` from an absolute path
  /// to a model asset bundle stored locally on the device and the default `LlmInference.Options`.
  ///
  /// - Parameters:
  ///   - modelPath: The absolute path to a model asset bundle stored locally on the device.
  @objc public convenience init(modelPath: String) {
    let options = Options(modelPath: modelPath)
    self.init(options: options)
  }

  /// Generates a response based on the input text.
  ///
  /// - Parameters:
  ///   - inputText: A `String` that is used to query the LLM.
  /// - Throws: An error if the LLM's response is invalid.
  @objc public func generateResponse(inputText: String) throws -> String {

    /// Disallow response generation if another response generation call is already in progress.
    try shouldContinueWithResponseGeneration()

    let tokens = try llmTaskRunner.predict(inputText: inputText)

    responseGenerationInProgress = false

    guard let humanReadableLlmResponse = LlmInference.humanReadableString(llmResponses: tokens)
    else {
      throw GenAiInferenceError.invalidResponse
    }

    return humanReadableLlmResponse
  }

  /// Generates a response based on the input text asynchronously. The `progress` callback returns
  /// the partial responses from the LLM or any errors. `completion` callback is invoked once the
  /// LLM is done generating responses.
  ///
  /// - Parameters:
  ///   - progress: A callback invoked when a partial response is available from the LLM.
  ///   - completion: A callback invoked when the LLM finishes response generation.
  /// - Throws: An error if the LLM's response is invalid.
  @objc public func generateResponseAsync(
    inputText: String,
    progress: @escaping (_ partialResponse: String?, _ error: Error?) -> Void,
    completion: @escaping (() -> Void)
  ) throws {
    /// Disallow response generation if another response generation call is already in progress.
    try shouldContinueWithResponseGeneration()

    /// Used to make a decision about whitespace stripping.
    var receivedFirstToken = true

    llmTaskRunner.predict(
      inputText: inputText,
      progress: { partialResponseStrings, error in

        guard let responseStrings = partialResponseStrings,
          let humanReadableLlmResponse = LlmInference.humanReadableString(
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

// Extension to `LlmInference` for defining `LlmInference.Options`
extension LlmInference {
  /// Options for setting up a `LlmInference`.
  ///
  /// Note: Inherits from `NSObject` for Objective C interoperability.
  @objc(MPPLLMInferenceOptions) public final class Options: NSObject {
    /// The absolute path to the model asset bundle stored locally on the device.
    @objc public var modelPath: String

    /// The total length of the kv-cache. In other words, this is the total number of input + output
    /// tokens the model needs to handle.
    @objc public var maxTokens: Int = 512

    /// The top K number of tokens to be sampled from for each decoding step. A value of 1 means
    /// greedy decoding. Defaults to 40.
    @objc public var topk: Int = 40

    /// The randomness when decoding the next token. A value of 0.0f means greedy decoding. Defaults
    /// to 0.8.
    @objc public var temperature: Float = 0.8

    /// The random seed for sampling tokens.
    @objc public var randomSeed: Int = 0

    /// Creates a new instance of `Options` with the modelPath and default values of
    /// `maxTokens`, `topK``, `temperature` and `randomSeed`.
    /// This function is only intended to be used from Objective C.
    ///
    /// - Parameters:
    ///   - modelPath: The absolute path to a model asset bundle stored locally on the device.
    @objc public init(modelPath: String) {
      self.modelPath = modelPath
      super.init()
    }
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
