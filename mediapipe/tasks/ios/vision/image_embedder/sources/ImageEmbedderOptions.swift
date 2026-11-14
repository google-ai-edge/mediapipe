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
package import third_party_mediapipe_tasks_c_vision_image_embedder_image_embedder

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksCore
  import MediaPipeTasksVisionCore
#endif

/// Configuration for an `ImageEmbedder` instance.
///
/// Not `Equatable` or `Hashable` because `liveStreamResultHandler` is a closure.
public struct ImageEmbedderOptions: Sendable {
  /// Asynchronous handler receiving live-stream output and frame timestamp in milliseconds.
  public typealias LiveStreamResultHandler =
    @Sendable (Result<ImageEmbedderResult, any Error>, _ timestampInMilliseconds: Int) -> Void

  /// Model asset and hardware accelerator settings.
  public var baseOptions: BaseOptions

  /// Execution mode selecting single image, sequential video, or asynchronous live stream.
  public var runningMode: VisionRunningMode

  /// Whether to normalize returned feature vectors with L2 norm.
  public var l2Normalize: Bool

  /// Whether to scalar-quantize returned embeddings to bytes.
  public var quantize: Bool

  /// Asynchronous handler required for `.liveStream` and disallowed for `.image` or `.video`.
  public var liveStreamResultHandler: LiveStreamResultHandler?

  /// Creates configuration with model, running mode, embedder settings, and optional stream handler.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    runningMode: VisionRunningMode = .image,
    l2Normalize: Bool = false,
    quantize: Bool = false,
    liveStreamResultHandler: LiveStreamResultHandler? = nil
  ) {
    self.baseOptions = baseOptions
    self.runningMode = runningMode
    self.l2Normalize = l2Normalize
    self.quantize = quantize
    self.liveStreamResultHandler = liveStreamResultHandler
  }

  package func withCOptions<R>(
    cCallback: MpImageEmbedderResultCallbackFn?,
    _ body: (inout MpImageEmbedderOptions) throws -> R
  ) rethrows -> R {
    let embedderOptions = EmbedderOptions(l2Normalize: l2Normalize, quantize: quantize)
    return try baseOptions.withCOptions { cBase in
      try embedderOptions.withCOptions { cEmbedder in
        var cOptions = MpImageEmbedderOptions(
          base_options: cBase,
          running_mode: runningMode.cRunningMode,
          embedder_options: cEmbedder,
          result_callback: cCallback
        )
        return try body(&cOptions)
      }
    }
  }
}
