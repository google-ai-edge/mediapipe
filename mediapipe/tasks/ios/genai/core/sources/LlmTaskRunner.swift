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
/// Note: Tasks should not attempt to clear undeleted caches on initialization since user can create
/// multiple instances of the task and there is now way of knowing whether they are still
/// active. Deleting caches of active task instances will result in crashes when the C++
/// functions are invoked.
/// Instead tasks can encapsulate `clearAllCachedFiles()` to provide a function to delete
/// any undeleted caches when the user wishes to.
final class LlmTaskRunner {
  private typealias CLlmSession = UnsafeMutableRawPointer

  private static let cacheSuffix = ".cache"
  private static let globalCacheDirectory = FileManager.default.temporaryDirectory
    .versionIndependentAppending(component: "mediapipe.genai.inference.cache")
  private static let cacheDirectory = LlmTaskRunner.globalCacheDirectory
    .versionIndependentAppending(component: "\(UUID().uuidString)")

  private let cLlmSession: CLlmSession

  private let modelCacheFile: URL

  /// Creates a new instance of `LlmTaskRunner` with the given session config.
  ///
  /// - Parameters:
  ///   - sessionConfig: C session config of type `LlmSessionConfig`.
  init(config: Config) throws {
    guard FileManager.default.fileExists(atPath: config.modelPath),
      let modelName = config.modelPath.components(separatedBy: "/").last
    else {
      throw GenAiInferenceError.modelNotFound
    }

    /// Adding a `UUID` prefix to the cache path to prevent the app from crashing if a model cache
    /// is already found in the temporary directory.
    /// Cache will be deleted when the task runner is de-allocated. Preferring deletion on
    /// de-allocation to deleting all caches on initialization to prevent model caches of
    /// other task runners from being de-allocated prematurely during their life time.
    ///
    /// Note: No safe guards for session creation since the C APIs only throw fatal errors.
    /// `LlmInferenceEngine_CreateSession()` will always return a llm session if the call
    /// completes.
    cLlmSession = LlmTaskRunner.cacheDirectory.path.withCString { cCacheDir in
      return config.modelPath.withCString { cModelPath in
        let cSessionConfig = LlmSessionConfig(
          model_path: cModelPath,
          cache_dir: cCacheDir,
          sequence_batch_size: Int(config.sequenceBatchSize),
          num_decode_steps_per_sync: Int(config.numberOfDecodeStepsPerSync),
          max_tokens: Int(config.maxTokens),
          topk: Int(config.topk),
          temperature: config.temperature,
          random_seed: config.randomSeed)
        return withUnsafePointer(to: cSessionConfig) { LlmInferenceEngine_CreateSession($0) }
      }
    }

    modelCacheFile = LlmTaskRunner.cacheDirectory.versionIndependentAppending(
      component: "\(modelName)\(LlmTaskRunner.cacheSuffix)")
  }

  /// Invokes the C inference engine with the given input text to generate an array of `String`
  /// responses from the LLM.
  ///
  /// - Parameters:
  ///   - inputText: A `String` that is used to query the LLM.
  /// - Throws: An error if the LLM's response is invalid.
  func predict(inputText: String) throws -> [String] {
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

  func predict(
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

  /// Clears all cached files created by `LlmInference` to prevent exponential growth of your app
  /// size. Please ensure that this method is not called during the lifetime of any instances of
  /// `LlmTaskRunner`.
  class func clearAllCachedFiles() {
    // Delete directory
    do { 
      try FileManager.default.removeItem(at: LlmTaskRunner.globalCacheDirectory)
      print("Success on deleting")
    }
    catch {
      print("Error in deleting")
      /// Errors thrown are not relevant to the user. They are usual not found errors.
    }
  }

  deinit {
    LlmInferenceEngine_Session_Delete(cLlmSession)
    
    /// Responsibly deleting the model cache.
    /// Performing on current thread since only one file needs to be deleted.
    ///
    /// Note: Implementation will have to be updated if C++ core changes the cache prefix.
    ///
    /// Note: `deinit` does not get invoked in the following circumstances:
    /// 1. If a crash occurs before the task runner is de-allocated.
    /// 2. If an instance of the task is created from `main()` and the app is terminated.
    ///    For eg:, if the task is an instance variable of the main `ViewController` which doesn't
    ///    get destroyed until the app quits.
    /// Task interfaces that use the task runner should additionally provide a function that
    /// encapsulates `LlmTaskrRunner.clearAllCachedFiles()` to cleanup any undeleted caches to
    /// avoid exponential growth in app size. OS clears these directories only if the device runs
    /// out of storage space.
    /// Tasks should not attempt to clear undeleted caches on initialization since user can create
    /// multiple instances of the task and there is now way of knowing whether they are still
    /// active. Deleting caches of active task instances will result in crashes when the C++
    /// functions are invoked.
    do {
      try FileManager.default.removeItem(at: modelCacheFile)
    } catch {
      // Could not delete file. Common cause: file not found.
    }
  }
}

extension LlmTaskRunner {
  /// Configuration for setting up a `LlmTaskRunner`.
  struct Config {
    /// The absolute path to the model asset bundle stored locally on the device.
    let modelPath: String

    let sequenceBatchSize: UInt

    let numberOfDecodeStepsPerSync: UInt

    /// The total length of the kv-cache. In other words, this is the total number of input + output
    /// tokens the model needs to handle.
    let maxTokens: UInt

    /// The top K number of tokens to be sampled from for each decoding step. A value of 1 means
    /// greedy decoding. Defaults to 40.
    let topk: UInt

    /// The randomness when decoding the next token. A value of 0.0f means greedy decoding. Defaults
    /// to 0.8.
    let temperature: Float

    /// The random seed for sampling tokens.
    let randomSeed: Int

    /// Creates a new instance of `Config` with the provided values.
    ///
    /// - Parameters:
    ///   - modelPath: The absolute path to a model asset bundle stored locally on the device.
    ///   - sequenceBatchSize: Sequence batch size for encoding. Used by GPU only. Number of
    /// input tokens to process at a time for batch processing. Setting this value to 1 means both
    /// the encoding and decoding share the same graph of sequence length of 1. Setting this value
    /// to 0 means the batch size will be optimized
    /// programmatically.
    ///   - numberOfDecodeStepsPerSync: Number of decode steps per sync. Used by GPU only.
    /// The default value is 3.
    ///   - maxTokens: Maximum number of tokens for input and output.
    ///   - topk: Top K number of tokens to be sampled from for each decoding step.
    ///   - temperature: Randomness when decoding the next token, 0.0f means greedy decoding.
    ///   - random_seed: Random seed for sampling tokens.
    init(
      modelPath: String, sequenceBatchSize: UInt, numberOfDecodeStepsPerSync: UInt, maxTokens: UInt,
      topk: UInt, temperature: Float, randomSeed: Int
    ) {
      self.modelPath = modelPath
      self.sequenceBatchSize = sequenceBatchSize
      self.numberOfDecodeStepsPerSync = numberOfDecodeStepsPerSync
      self.maxTokens = maxTokens
      self.topk = topk
      self.temperature = temperature
      self.randomSeed = randomSeed
    }
  }
}

private extension LlmTaskRunner {
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

private extension LlmTaskRunner {
  class func responseStrings(from responseContext: LlmResponseContext) -> [String]? {
    guard let cResponseArray = responseContext.response_array else {
      return nil
    }

    var responseStrings: [String] = []
    for responseIndex in 0..<Int(responseContext.response_count) {
      /// Throw an error if the response string is `NULL`.
      guard let cResponseString = cResponseArray[responseIndex] else {
        return nil
      }
      responseStrings.append(String(cString: cResponseString))
    }

    return responseStrings
  }
}

fileprivate extension URL {
  func versionIndependentAppending(component: String) -> URL {
    if #available(iOS 16, *) {
      return self.appending(component: component)
    } else {
      return self.appendingPathComponent(component)
    }
  }
}
