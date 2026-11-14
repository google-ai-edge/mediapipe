// Copyright 2026 The MediaPipe Authors.
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
package import third_party_mediapipe_tasks_c_text_text_proofreader_text_proofreader_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
#endif

/// Configuration options for the MediaPipe `TextProofreader` task.
public struct TextProofreaderOptions: Sendable, Equatable, Hashable {
  private static let defaultMinLogSeverity: Int32 = 4

  /// Model asset and hardware delegate configuration.
  public var baseOptions: BaseOptions

  /// Maximum combined input and output token budget, or `0` to use the model's default capacity.
  public var maxNumTokens: Int

  /// Directory used for weight and program caches, or `":nocache"` to disable caching.
  public var cacheDirectory: String?

  /// Creates options with the specified model configuration, token limit, and cache directory.
  public init(
    baseOptions: BaseOptions = BaseOptions(), maxNumTokens: Int = 0, cacheDirectory: String? = nil
  ) {
    self.baseOptions = baseOptions
    self.maxNumTokens = maxNumTokens
    self.cacheDirectory = cacheDirectory
  }

  /// Invokes `body` with a C `MpTextProofreaderOptions` struct whose pointer fields remain valid
  /// for the duration of the closure call.
  package func withCOptions<Result>(
    _ body: (MpTextProofreaderOptions) throws -> Result
  ) throws -> Result {
    guard baseOptions.delegate == .cpu else {
      throw TasksError.invalidArgument(
        "INVALID_ARGUMENT: TextProofreader only supports the CPU delegate.")
    }
    guard maxNumTokens >= 0, let cMaxTokens = Int32(exactly: maxNumTokens) else {
      throw TasksError.invalidArgument(
        "INVALID_ARGUMENT: maxNumTokens (\(maxNumTokens)) must be between 0 and \(Int32.max).")
    }
    return try baseOptions.withCOptions { cBaseOptions in
      try cacheDirectory.withCString { cacheDirPtr in
        var cOptions = MpTextProofreaderOptions()
        cOptions.base_options = cBaseOptions
        cOptions.max_num_tokens = cMaxTokens
        cOptions.cache_dir = cacheDirPtr
        cOptions.min_log_severity = Self.defaultMinLogSeverity
        return try body(cOptions)
      }
    }
  }
}
