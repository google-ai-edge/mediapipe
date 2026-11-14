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
package import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_vision_core_image
package import third_party_mediapipe_tasks_c_vision_core_image_processing_options
package import third_party_mediapipe_tasks_c_vision_image_classifier_image_classifier_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Classifies objects or scenes in still images, video frames, and live camera streams.
///
/// - Note: Instances are not thread-safe; serialize concurrent access (for example, in an `actor`).
public final class ImageClassifier {
  private let cClassifier: MpImageClassifierPtr
  private let runner: VisionTaskRunner
  private let liveStreamRegistration: VisionLiveStreamRegistration?

  /// Initializes a classifier configured with `options`.
  ///
  /// - Throws: `TasksError` if option validation or underlying MediaPipe graph creation fails.
  public init(options: ImageClassifierOptions) throws {
    let handler = options.liveStreamResultHandler
    let runner = try VisionTaskRunner(
      baseOptions: options.baseOptions,
      runningMode: options.runningMode,
      roiAllowed: true,
      normRectAllowed: true,
      hasLiveStreamHandler: handler != nil
    )

    var registration: VisionLiveStreamRegistration?
    var cCallback: MpImageClassifierResultCallback?
    if options.runningMode == .liveStream, let handler {
      let registered = try VisionLiveStreamRegistration.register(
        queueLabelSuffix: "image_classifier",
        resultType: MpImageClassifierResult.self,
        as: MpImageClassifierResultCallback.self,
        decode: { cResultPtr, _, timestampMs in
          ImageClassifierResult(cResult: cResultPtr.pointee, timestampInMilliseconds: timestampMs)
        },
        handler: handler
      )
      registration = registered.registration
      cCallback = registered.cCallback
    }

    do {
      self.cClassifier = try options.withCOptions(cCallback: cCallback) { cOptions in
        var classifierPtr: MpImageClassifierPtr?
        try MpStatus.check(defaultMessage: "Failed to create ImageClassifier.") { errorMessage in
          MpImageClassifierCreate(&cOptions, &classifierPtr, &errorMessage)
        }
        guard let validClassifier = classifierPtr else {
          throw TasksError.internalError("MpImageClassifierCreate returned a null pointer.")
        }
        return validClassifier
      }
      self.runner = runner
      self.liveStreamRegistration = registration
    } catch {
      registration?.unregister()
      throw error
    }
  }

  /// Initializes a classifier in `.image` mode from a model file path with default options.
  public convenience init(modelPath: String) throws {
    try self.init(
      options: ImageClassifierOptions(baseOptions: BaseOptions(modelAssetPath: modelPath))
    )
  }

  deinit {
    _ = MpImageClassifierClose(cClassifier, nil)
    liveStreamRegistration?.unregister()
  }

  /// Classifies a still `image` when running in `.image` mode.
  public func classify(
    image: MPImage,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> ImageClassifierResult {
    try runner.withPreparedInput(
      image: image,
      imageProcessingOptions: imageProcessingOptions,
      expectedMode: .image
    ) { cImage, cOpts in
      var cResult = MpImageClassifierResult()
      try MpStatus.check(defaultMessage: "Failed to classify image.") { errorMessage in
        MpImageClassifierClassifyImage(cClassifier, cImage, cOpts, &cResult, &errorMessage)
      }
      defer { MpImageClassifierCloseResult(&cResult) }
      return ImageClassifierResult(cResult: cResult, timestampInMilliseconds: 0)
    }
  }

  /// Classifies `videoFrame` at monotonically increasing `timestampInMilliseconds` in `.video` mode.
  public func classify(
    videoFrame: MPImage,
    timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> ImageClassifierResult {
    try runner.withPreparedInput(
      image: videoFrame,
      imageProcessingOptions: imageProcessingOptions,
      expectedMode: .video
    ) { cImage, cOpts in
      var cResult = MpImageClassifierResult()
      let ts = try runner.cTimestamp(timestampInMilliseconds)
      try MpStatus.check(defaultMessage: "Failed to classify video frame.") { errorMessage in
        MpImageClassifierClassifyForVideo(cClassifier, cImage, cOpts, ts, &cResult, &errorMessage)
      }
      defer { MpImageClassifierCloseResult(&cResult) }
      return ImageClassifierResult(
        cResult: cResult, timestampInMilliseconds: timestampInMilliseconds)
    }
  }

  /// Enqueues `image` at monotonically increasing `timestampInMilliseconds` in `.liveStream` mode.
  public func classifyAsync(
    image: MPImage,
    timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws {
    try runner.withPreparedInput(
      image: image,
      imageProcessingOptions: imageProcessingOptions,
      expectedMode: .liveStream
    ) { cImage, cOpts in
      let ts = try runner.cTimestamp(timestampInMilliseconds)
      try MpStatus.check(defaultMessage: "Failed to classify live-stream image.") { errorMessage in
        MpImageClassifierClassifyAsync(cClassifier, cImage, cOpts, ts, &errorMessage)
      }
    }
  }
}
