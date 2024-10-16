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
/// An instance of `LlmInference` will only be deallocated after all sessions created from it are
/// destroyed. This means that an LLM inference can stay in memory even if a reference to it goes
/// out of scope if at least one of its sessions outlives its scope.
///
/// Note: Inherits from `NSObject` for Objective C interoperability.
///
/// Note: Initializing an LLM inference engine is an expensive operation. Avoid initializing it on
/// the main thread.
@objc(MPPLLMInference) public final class LlmInference: NSObject {
  private static let numberOfDecodeStepsPerSync = 3
  private static let sequenceBatchSize = 0
  private static let responseGenerationInProgressQueueName =
    "com.google.mediapipe.genai.isResponseGenerationInProgressQueue"

  /// Provides key metrics including initialization duration.
  public private(set) var metrics: Metrics

  private let llmTaskRunner: LlmTaskRunner

  /// Serial queue that reads and updates `responseGenerationInProgress` to restrict simultaneous
  /// execution of response generation functions across sessions created from this
  /// `LlmInference`.
  private let responseGenerationInProgressQueue = DispatchQueue(
    label: LlmInference.responseGenerationInProgressQueueName)

  /// Tracks whether a response generation is in progress.
  private var responseGenerationInProgress = false

  /// Creates a new instance of `LlmInference` with the given options.
  /// An instance of `LlmInference` will only be deallocated after all sessions created from it are
  /// destroyed. This means that an LLM inference can stay in memory even if the reference to it
  /// goes out of scope if at least one of its sessions outlives its scope.
  ///
  /// - Parameters:
  ///   - options: The options of type `LlmInference.Options` to use for configuring the
  /// `LlmInference`.
  /// - Throws: An error if `LlmInference` instance could not be initialized.
  @objc public init(options: Options) throws {
    let cacheDirectory = FileManager.default.temporaryDirectory.path

    let sequenceBatchSize = LlmInference.sequenceBatchSize
    let numberOfDecodeStepsPerSync = LlmInference.numberOfDecodeStepsPerSync
    let timeBeforeInit = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    llmTaskRunner = try options.modelPath.withCString { modelPath in
      try cacheDirectory.withCString { cacheDirectory in
        try options.supportedLoraRanks.withUnsafeMutableBufferPointer { supportedLoraRanks in
          let modelSetting = LlmModelSettings(
            model_path: modelPath,
            cache_dir: cacheDirectory,
            max_num_tokens: options.maxTokens,
            num_decode_steps_per_sync: numberOfDecodeStepsPerSync,
            sequence_batch_size: sequenceBatchSize,
            number_of_supported_lora_ranks: supportedLoraRanks.count,
            supported_lora_ranks: supportedLoraRanks.baseAddress,
            max_top_k: options.maxTopk,
            llm_activation_data_type: kLlmActivationDataTypeDefault,
            num_draft_tokens: 0,
            wait_for_weight_uploads: options.waitForWeightUploads)
          return try LlmTaskRunner(modelSettings: modelSetting)
        }
      }
    }
    let timeAfterInit = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    metrics = Metrics(
      initializationTimeInMillis: (TimeInterval(timeAfterInit - timeBeforeInit) * 1000)
        / TimeInterval(NSEC_PER_SEC))

    super.init()
  }

  /// A convenience initializer that creates a new instance of `LlmInference` from an absolute path
  /// to a model asset bundle stored locally on the device and the default `LlmInference.Options`.
  /// An instance of `LlmInference` will only be deallocated after all sessions created from it are
  /// destroyed. This means that an LLM inference can stay in memory even if the reference to it
  /// goes out of scope if at least one of its sessions outlives its scope.
  ///
  /// - Parameters:
  ///   - modelPath: The absolute path to a model asset bundle stored locally on the device.
  /// - Throws: An error if `LlmInference` instance could not be initialized.
  @objc public convenience init(modelPath: String) throws {
    let options = Options(modelPath: modelPath)
    try self.init(options: options)
  }

  /// Creates and returns a session runner that wraps around a new session created by the underlying
  /// LLM engine.
  ///
  /// - Parameters:
  ///   - sessionConfig: The C config of type `LlmSessionConfig` that configures how to execute the
  /// model.
  /// - Returns:
  ///   - An `LlmSessionRunner` that wraps around a new session.
  /// - Throws: An error if the underlying engine could not create a session.
  func createSessionRunner(sessionConfig: LlmSessionConfig) throws -> LlmSessionRunner {
    return try llmTaskRunner.createSessionRunner(sessionConfig: sessionConfig)
  }

  /// Generates a response based on the input text. This function creates a new session for each
  /// call. If you want to have a stateful inference, use `LlmInference.Session`'s
  /// `generateResponse()` instead.
  ///
  /// - Parameters:
  ///   - inputText: A `String` that is used to query the LLM.
  /// - Throws: An error if the LLM's response is invalid.
  @objc public func generateResponse(inputText: String) throws -> String {
    let session = try LlmInference.Session(llmInference: self)
    try session.addQueryChunk(inputText: inputText)
    return try session.generateResponse()
  }

  /// Generates a response based on the input text asynchronously. The `progress` callback returns
  /// the partial responses from the LLM or any errors. `completion` callback is invoked once the
  /// LLM is done generating responses. This function creates a new session for each call.
  /// If you want to have a stateful inference, use `LlmInference.Session`'s
  /// `generateResponseAsync(progress: completion:) throws` instead.
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
    let session = try LlmInference.Session(llmInference: self)
    try session.addQueryChunk(inputText: inputText)
    try session.generateResponseAsync(progress: progress, completion: completion)
  }

  /// Generates a response based on the input text asynchronously. This function creates a new
  /// session for each call. If you want to have a stateful inference, use `LlmInference.Session`'s
  /// `generateResponseAsync() -> AsyncThrowingStream<String, Error>` instead.
  ///
  /// - Parameters:
  ///   - inputText: The prompt used to query the LLM.
  /// - Returns: An async throwing stream that contains the partial responses from the LLM.
  @available(iOS 13, macOS 10.15, tvOS 13, watchOS 6, *)
  public func generateResponseAsync(inputText: String) -> AsyncThrowingStream<String, Error> {
    AsyncThrowingStream { continuation in
      do {
        let session = try LlmInference.Session(llmInference: self)
        try session.addQueryChunk(inputText: inputText)
        try session.generateResponseAsync(
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

  /// If no response generation using any session created from this `LlmInference` is currently in
  /// progress, this function updates the response generation state to `true` and returns
  /// successfully thereby granting access to its caller to execute response generation.
  /// If this function throws an error, the invoking session must abort the response generation
  /// call. This function must be called before invoking the response generation function on the
  /// underlying `LlmSessionRunner`.
  ///
  /// - Throws: An error if response generation is already in progress.
  func shouldContinueWithResponseGeneration() throws {
    /// `responseGenerationInProgressQueue` is a serial queue. Executing a sync block on a serial
    /// queue ensures that at any time only one call to this function tests and writes the current
    /// state of response generation. All other calls are blocked until the state is
    /// updated. If the state indicates that response generation is currently in progress, the
    /// block throws an error. Since it is a synchronous block that blocks execution until it is
    /// complete, the error is in turn propagated as an error thrown by the function.
    try responseGenerationInProgressQueue.sync {
      if !responseGenerationInProgress {
        responseGenerationInProgress = true
      } else {
        throw GenAiInferenceError.illegalMethodCall
      }
    }
  }

  /// Marks response generation as complete by updating the state to `false`. Any session created
  /// using this `LlmInference` must use this function to indicate the completion of response
  /// generation using the underlying `LlmSessionRunner`.
  func markResponseGenerationCompleted() {
    responseGenerationInProgressQueue.sync {
      responseGenerationInProgress = false
    }
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

    /// Maximum top k, which is the max Top-K value supported for all sessions created with the
    /// `LlmInference`, used by GPU only. If a session with Top-K value larger than this is being
    /// asked to be created, it will be rejected(throw error). A value of 1 means only greedy
    // decoding is supported for any sessions created with this `LlmInference`. Default value is 40.
    @objc public var maxTopk: Int = 40

    /// The supported lora ranks for the base model. Used by GPU only.
    @objc public var supportedLoraRanks: [Int] = []

    /// If true, waits for weights to finish uploading when initializing. Otherwise initialization
    /// may finish before weights have finished uploading which might push some of the weight upload
    /// time into input processing.
    @objc public var waitForWeightUploads: Bool = false

    /// Creates a new instance of `Options` with the given `modelPath` and default values of
    /// `maxTokens`, `maxTopk`, `supportedLoraRanks`.
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

extension LlmInference {
  /// Provides some key metrics for the `LlmInference`.
  ///
  /// Note: Inherits from `NSObject` for Objective C interoperability.
  @objc(MPPLLMInferenceMetrics) public final class Metrics: NSObject {
    /// The time it took to initialize the LLM inference engine, in milliseconds.
    /// If you want to include the time it took to load the model weights, set
    /// `LlmInference.Options.waitForWeightUploads` to true.
    @objc public private(set) var initializationTimeInMillis: TimeInterval

    @objc public init(initializationTimeInMillis: TimeInterval) {
      self.initializationTimeInMillis = initializationTimeInMillis
    }
  }
}
