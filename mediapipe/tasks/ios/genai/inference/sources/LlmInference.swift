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
/// An instance of LLM Inferece will only be deallocated after all sessions created from it are 
/// destroyed. This means that an LLM inference can stay in memory even if a reference to it goes 
/// out of scope if atleast one of its sessions outlive its scope.
///
/// Note: Inherits from `NSObject` for Objective C interoperability.
@objc(MPPLLMInference) public final class LlmInference: NSObject {
  private static let numberOfDecodeStepsPerSync = 3
  private static let sequenceBatchSize = 0
  private static let responseGenerationInProgressQueueName =
    "com.google.mediapipe.genai.isResponseGenerationInProgressQueue"

  private let llmTaskRunner: LlmTaskRunner

  // Queue that restricts access to the response generation functions simultaneously.
  private let responseGenerationInProgressQueue = DispatchQueue(
    label: LlmInference.responseGenerationInProgressQueueName)

  /// Tracks whether a response generation is in progress.
  /// Readers writers lock to prevent race condition as this variable can be accessed from multiple
  /// threads.
  private var responseGenerationInProgress = false

  /// Creates a new instance of `LlmInference` with the given options.
  ///
  /// - Parameters:
  ///   - options: The options of type `LlmInference.Options` to use for configuring the
  /// `LlmInference`.
  /// - Throws: An error if `LlmInference` instance could not be initialized.
  @objc public init(options: Options) throws {
    let cacheDirectory = FileManager.default.temporaryDirectory.path

    llmTaskRunner = try options.modelPath.withCString { modelPath in
      try cacheDirectory.withCString { cacheDirectory in
        try options.supportedLoraRanks.withUnsafeMutableBufferPointer { supportedLoraRanks in
          let modelSetting = LlmModelSettings(
            model_path: modelPath,
            cache_dir: cacheDirectory,
            max_num_tokens: options.maxTokens,
            num_decode_steps_per_sync: LlmInference.numberOfDecodeStepsPerSync,
            sequence_batch_size: LlmInference.sequenceBatchSize,
            number_of_supported_lora_ranks: options.supportedLoraRanks.count,
            supported_lora_ranks: supportedLoraRanks.baseAddress,
            max_top_k: options.maxTopk)
          return try LlmTaskRunner(modelSettings: modelSetting)
        }
      }
    }

    super.init()
  }

  /// A convenience initializer that creates a new instance of `LlmInference` from an absolute path
  /// to a model asset bundle stored locally on the device and the default `LlmInference.Options`.
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
    let llmSessionRunner = try llmTaskRunner.createSessionRunner(sessionConfig: sessionConfig)
    return llmSessionRunner
  }

  /// If no response generation using this `llmInference` is currently in progress, this function
  /// updates the response generation state and returns successfully granting access to its caller
  /// to execute response generation. If response generation is already in progress, throws an
  /// error.
  /// Any
  func shouldContinueWithResponseGeneration() throws {
    /// `responseGenerationInProgressQueue` is a serial queue. executing a sync block on a serial
    /// queue ensures that at any time only one call to this function tests and writes the current
    /// state of response generation. All other calls are blocked until the state is
    /// updated. If the state indicates that response generation is currently in progress, the
    /// block throws an error. Since it is a synchronous block that blocks execution until it is
    /// complete, the error is in turn propogated as an error thrown by the function.
    try responseGenerationInProgressQueue.sync {
      if !responseGenerationInProgress {
        responseGenerationInProgress = true
      } else {
        throw GenAiInferenceError.illegalMethodCall
      }
    }
  }

  /// Marks response generation as complete by updating the state. Any session created using this
  /// `llmInference` must use this function to indicate that the call to the underlying C session
  /// for response generation is completed.
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
    /// engine, used by GPU only. If a session with Top-K value larger than this is being asked to
    /// be created, it will be rejected(throw error). If not provided, the max top k will be 1,
    /// which means only greedy decoding is supported for any sessions created with this
    /// `LlmInference``.
    @objc public var maxTopk: Int = 40

    /// The supported lora ranks for the base model. Used by GPU only.
    @objc public var supportedLoraRanks: [Int] = []

    /// Creates a new instance of `Options` with the given `modelPath` and default values of
    /// `maxTokens`, `maxTopk` and `supportedLoraRanks`.
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
