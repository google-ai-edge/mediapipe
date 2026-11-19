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
private import third_party_mediapipe_tasks_c_core_mp_status
private import third_party_mediapipe_tasks_c_vision_image_embedder_image_embedder

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Extracts dense numerical feature vectors from images, video frames, or live camera streams.
///
/// - Important: Instances are not thread-safe (`Sendable` is intentionally omitted);
///   serialize `embed`, `embedAsync`, and `close` calls from a single execution context.
public final class ImageEmbedder {
  private let runner: VisionTaskRunner
  private let lock = NSLock()
  private var cEmbedder: MpImageEmbedderPtr?
  /// Receives the live-stream results. The C task keeps an unretained pointer to it until the task
  /// has been closed. Guarded by `lock` and released together with `cEmbedder`, so that a closed
  /// embedder no longer retains the result handler.
  private var liveStreamContext: VisionLiveStreamContext<MpImageEmbedderResult>?

  /// Creates an instance from `options`.
  public init(options: ImageEmbedderOptions) throws {
    let runner = try VisionTaskRunner(
      baseOptions: options.baseOptions,
      runningMode: options.runningMode,
      roiAllowed: true,
      normRectAllowed: true,
      hasLiveStreamHandler: options.liveStreamResultHandler != nil
    )

    let liveStreamContext = options.liveStreamResultHandler.map { handler in
      VisionLiveStreamContext<MpImageEmbedderResult>(
        queueLabelSuffix: "imageEmbedder",
        decode: { cResultPtr, _, ts in
          ImageEmbedderResult(cResult: cResultPtr.pointee, timestampInMilliseconds: ts)
        },
        handler: handler
      )
    }

    var created: MpImageEmbedderPtr?
    try options.withCOptions(liveStreamContext: liveStreamContext) { cOptions in
      try MpStatus.check { MpImageEmbedderCreate(&cOptions, &created, &$0) }
    }
    guard let created else {
      throw TasksError.internalError("Failed to create MpImageEmbedder instance.")
    }

    self.runner = runner
    self.cEmbedder = created
    self.liveStreamContext = liveStreamContext
  }

  /// Creates an `.image` mode instance from a local model file.
  public convenience init(modelPath: String) throws {
    try self.init(
      options: ImageEmbedderOptions(baseOptions: BaseOptions(modelAssetPath: modelPath)))
  }

  deinit {
    let (embedder, liveStreamContext) = takeEmbedderToClose()
    if let embedder {
      // The C task calls the live-stream context until `MpImageEmbedderClose` has returned.
      withExtendedLifetime(liveStreamContext) { _ = MpImageEmbedderClose(embedder, nil) }
    }
  }

  /// Runs synchronous embedding extraction on a standalone image in `.image` mode.
  public func embed(
    image: MPImage, imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> ImageEmbedderResult {
    let embedder = try activeEmbedder()
    return try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .image
    ) { cImage, cProcessing in
      var cResult = MpImageEmbedderResult()
      try MpStatus.check { MpImageEmbedderEmbedImage(embedder, cImage, cProcessing, &cResult, &$0) }
      defer { MpImageEmbedderCloseResult(&cResult) }
      return ImageEmbedderResult(cResult: cResult)
    }
  }

  /// Runs synchronous embedding extraction on a sequential `videoFrame` in `.video` mode.
  public func embed(
    videoFrame: MPImage,
    timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> ImageEmbedderResult {
    let embedder = try activeEmbedder()
    let cTimestamp = try runner.cTimestamp(timestampInMilliseconds)
    return try runner.withPreparedInput(
      image: videoFrame, imageProcessingOptions: imageProcessingOptions, expectedMode: .video
    ) { cImage, cProcessing in
      var cResult = MpImageEmbedderResult()
      try MpStatus.check {
        MpImageEmbedderEmbedForVideo(embedder, cImage, cProcessing, cTimestamp, &cResult, &$0)
      }
      defer { MpImageEmbedderCloseResult(&cResult) }
      return ImageEmbedderResult(cResult: cResult, timestampInMilliseconds: timestampInMilliseconds)
    }
  }

  /// Enqueues a live-stream frame tagged with monotonically increasing milliseconds.
  public func embedAsync(
    image: MPImage,
    timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws {
    let embedder = try activeEmbedder()
    let cTimestamp = try runner.cTimestamp(timestampInMilliseconds)
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .liveStream
    ) { cImage, cProcessing in
      try MpStatus.check {
        MpImageEmbedderEmbedAsync(embedder, cImage, cProcessing, cTimestamp, &$0)
      }
    }
  }

  /// Flushes pending live-stream packets and shuts down the underlying MediaPipe graph. Releases
  /// the live-stream result handler once the results that are still pending have been delivered.
  public func close() throws {
    let (embedder, liveStreamContext) = takeEmbedderToClose()
    guard let embedder else { return }
    // The C task calls the live-stream context until `MpImageEmbedderClose` has returned.
    try withExtendedLifetime(liveStreamContext) {
      try MpStatus.check(defaultMessage: "Failed to close ImageEmbedder.") {
        MpImageEmbedderClose(embedder, &$0)
      }
    }
  }

  /// Computes cosine similarity in `[-1.0, 1.0]` between two compatible embeddings.
  public static func cosineSimilarity(
    embedding1: Embedding,
    embedding2: Embedding
  ) throws -> Double {
    try CosineSimilarity.compute(between: embedding1, and: embedding2)
  }

  private func activeEmbedder() throws -> MpImageEmbedderPtr {
    guard let embedder = lock.withLock({ cEmbedder }) else {
      throw TasksError.invalidArgument("Task runner is currently not running.")
    }
    return embedder
  }

  /// Removes the C embedder and its live-stream context from this instance. The caller closes the
  /// embedder and keeps the context alive until the close call has returned.
  private func takeEmbedderToClose() -> (
    MpImageEmbedderPtr?, VisionLiveStreamContext<MpImageEmbedderResult>?
  ) {
    lock.withLock {
      defer {
        cEmbedder = nil
        liveStreamContext = nil
      }
      return (cEmbedder, liveStreamContext)
    }
  }
}
