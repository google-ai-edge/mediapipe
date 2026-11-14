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
package import third_party_mediapipe_tasks_c_vision_object_detector_object_detector_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Detects objects and their pixel-space bounding boxes across images, video frames, and live
/// camera streams.
///
/// - Note: `ObjectDetector` is not thread-safe. Synchronize concurrent calls externally across
///   threads.
public final class ObjectDetector {
  private let cDetector: MpObjectDetectorPtr
  private let runner: VisionTaskRunner
  private let liveStreamRegistration: VisionLiveStreamRegistration?

  /// Creates an `ObjectDetector` configured with `options`.
  public init(options: ObjectDetectorOptions) throws {
    let runner = try VisionTaskRunner(
      baseOptions: options.baseOptions,
      runningMode: options.runningMode,
      roiAllowed: false,
      normRectAllowed: true,
      hasLiveStreamHandler: options.liveStreamResultHandler != nil
    )
    var registration: VisionLiveStreamRegistration?
    var cCallback: MpObjectDetectorResultCallback?
    if let handler = options.liveStreamResultHandler {
      let registered = try VisionLiveStreamRegistration.register(
        queueLabelSuffix: "objectDetector",
        resultType: MpObjectDetectorResult.self,
        as: MpObjectDetectorResultCallback.self,
        decode: { cResultPtr, _, timestampMs in
          ObjectDetectorResult(cResult: cResultPtr.pointee, timestampInMilliseconds: timestampMs)
        },
        handler: handler
      )
      registration = registered.registration
      cCallback = registered.cCallback
    }

    var succeeded = false
    defer {
      if !succeeded {
        registration?.unregister()
      }
    }
    var detectorPtr: MpObjectDetectorPtr?
    try options.withCOptions(cCallback: cCallback) { cOptionsPtr in
      try MpStatus.check(defaultMessage: "Failed to create ObjectDetector.") { errorMsg in
        MpObjectDetectorCreate(cOptionsPtr, &detectorPtr, &errorMsg)
      }
    }
    guard let detectorPtr else {
      throw TasksError.internalError("MpObjectDetectorCreate succeeded without returning a task.")
    }
    succeeded = true
    self.cDetector = detectorPtr
    self.runner = runner
    self.liveStreamRegistration = registration
  }

  /// Creates an `ObjectDetector` in `.image` mode using the model file at `modelPath`.
  public convenience init(modelPath: String) throws {
    try self.init(
      options: ObjectDetectorOptions(baseOptions: BaseOptions(modelAssetPath: modelPath)))
  }

  deinit {
    _ = MpObjectDetectorClose(cDetector, nil)
    liveStreamRegistration?.unregister()
  }

  /// Detects objects in a standalone `image` (`.image` running mode only).
  public func detect(
    image: MPImage,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> ObjectDetectorResult {
    try runner.withPreparedInput(
      image: image,
      imageProcessingOptions: imageProcessingOptions,
      expectedMode: .image
    ) { cImage, cProcessingOptions in
      var cResult = MpObjectDetectorResult()
      defer { MpObjectDetectorCloseResult(&cResult) }
      try MpStatus.check(defaultMessage: "Failed to detect objects in image.") { errorMsg in
        MpObjectDetectorDetectImage(cDetector, cImage, cProcessingOptions, &cResult, &errorMsg)
      }
      return ObjectDetectorResult(cResult: cResult, timestampInMilliseconds: 0)
    }
  }

  /// Detects objects in a sequential `videoFrame` at monotonically increasing
  /// `timestampInMilliseconds` (`.video` running mode only).
  public func detect(
    videoFrame: MPImage,
    timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> ObjectDetectorResult {
    try runner.withPreparedInput(
      image: videoFrame,
      imageProcessingOptions: imageProcessingOptions,
      expectedMode: .video
    ) { cImage, cProcessingOptions in
      let cTimestamp = try runner.cTimestamp(timestampInMilliseconds)
      var cResult = MpObjectDetectorResult()
      defer { MpObjectDetectorCloseResult(&cResult) }
      try MpStatus.check(defaultMessage: "Failed to detect objects in video frame.") { errorMsg in
        MpObjectDetectorDetectForVideo(
          cDetector, cImage, cProcessingOptions, cTimestamp, &cResult, &errorMsg)
      }
      return ObjectDetectorResult(
        cResult: cResult, timestampInMilliseconds: timestampInMilliseconds)
    }
  }

  /// Submits a live camera `image` at monotonically increasing `timestampInMilliseconds` for
  /// asynchronous result delivery (`.liveStream` running mode only).
  public func detectAsync(
    image: MPImage,
    timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws {
    try runner.withPreparedInput(
      image: image,
      imageProcessingOptions: imageProcessingOptions,
      expectedMode: .liveStream
    ) { cImage, cProcessingOptions in
      let cTimestamp = try runner.cTimestamp(timestampInMilliseconds)
      try MpStatus.check(defaultMessage: "Failed to send live-stream frame to ObjectDetector.") {
        errorMsg in
        MpObjectDetectorDetectAsync(cDetector, cImage, cProcessingOptions, cTimestamp, &errorMsg)
      }
    }
  }
}
