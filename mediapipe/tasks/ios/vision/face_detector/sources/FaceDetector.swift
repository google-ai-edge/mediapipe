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
private import third_party_mediapipe_tasks_c_vision_face_detector_face_detector_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Detects human faces and facial keypoints in images, video frames, and live camera streams.
///
/// - Important: `FaceDetector` is **not** thread-safe. Synchronize access externally when calling
///   detection methods from multiple threads or tasks.
public final class FaceDetector {
  private let cDetector: MpFaceDetectorPtr
  private let runner: VisionTaskRunner
  private let liveStreamRegistration: VisionLiveStreamRegistration?

  /// Creates a `FaceDetector` configured with `options`.
  public init(options: FaceDetectorOptions) throws {
    let runner = try VisionTaskRunner(
      baseOptions: options.baseOptions,
      runningMode: options.runningMode,
      roiAllowed: false,
      normRectAllowed: true,
      hasLiveStreamHandler: options.liveStreamResultHandler != nil
    )

    let registration: VisionLiveStreamRegistration?
    let cCallback: MpFaceDetectorResultCallback?
    if options.runningMode == .liveStream, let handler = options.liveStreamResultHandler {
      let registered = try VisionLiveStreamRegistration.register(
        queueLabelSuffix: "faceDetector",
        resultType: MpFaceDetectorResult.self,
        as: MpFaceDetectorResultCallback.self,
        decode: { cResultPtr, _, timestampMs in
          FaceDetectorResult(cResult: cResultPtr.pointee, timestampInMilliseconds: timestampMs)
        },
        handler: handler
      )
      registration = registered.registration
      cCallback = registered.cCallback
    } else {
      registration = nil
      cCallback = nil
    }

    do {
      self.cDetector = try options.withCOptions(resultCallback: cCallback) { cOptions in
        var detectorPtr: MpFaceDetectorPtr?
        try MpStatus.check(defaultMessage: "Failed to create FaceDetector.") { errorMessage in
          MpFaceDetectorCreate(&cOptions, &detectorPtr, &errorMessage)
        }
        guard let detectorPtr else {
          throw TasksError.internalError("MpFaceDetectorCreate returned a null handle.")
        }
        return detectorPtr
      }
    } catch {
      registration?.unregister()
      throw error
    }
    self.runner = runner
    self.liveStreamRegistration = registration
  }

  /// Creates an `.image`-mode `FaceDetector` from a local model file path.
  public convenience init(modelPath: String) throws {
    try self.init(options: FaceDetectorOptions(baseOptions: BaseOptions(modelAssetPath: modelPath)))
  }

  deinit {
    _ = MpFaceDetectorClose(cDetector, nil)
    liveStreamRegistration?.unregister()
  }

  /// Detects faces in a standalone image when initialized in `.image` mode.
  public func detect(
    image: MPImage,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> FaceDetectorResult {
    try runner.withPreparedInput(
      image: image,
      imageProcessingOptions: imageProcessingOptions,
      expectedMode: .image
    ) { cImage, cOptionsPtr in
      var cResult = MpFaceDetectorResult()
      defer { MpFaceDetectorCloseResult(&cResult) }
      try MpStatus.check(defaultMessage: "Failed to detect faces in image.") { errorMessage in
        MpFaceDetectorDetectImage(cDetector, cImage, cOptionsPtr, &cResult, &errorMessage)
      }
      return FaceDetectorResult(cResult: cResult)
    }
  }

  /// Detects faces in a sequential video frame at monotonically increasing
  /// `timestampInMilliseconds` when initialized in `.video` mode.
  public func detect(
    videoFrame: MPImage,
    timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> FaceDetectorResult {
    let cTimestamp = try runner.cTimestamp(timestampInMilliseconds)
    return try runner.withPreparedInput(
      image: videoFrame,
      imageProcessingOptions: imageProcessingOptions,
      expectedMode: .video
    ) { cImage, cOptionsPtr in
      var cResult = MpFaceDetectorResult()
      defer { MpFaceDetectorCloseResult(&cResult) }
      try MpStatus.check(defaultMessage: "Failed to detect faces in video frame.") { errorMessage in
        MpFaceDetectorDetectForVideo(
          cDetector, cImage, cOptionsPtr, cTimestamp, &cResult, &errorMessage
        )
      }
      return FaceDetectorResult(cResult: cResult, timestampInMilliseconds: timestampInMilliseconds)
    }
  }

  /// Submits a live-stream frame at monotonically increasing `timestampInMilliseconds` for
  /// asynchronous detection when initialized in `.liveStream` mode.
  public func detectAsync(
    image: MPImage,
    timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws {
    let cTimestamp = try runner.cTimestamp(timestampInMilliseconds)
    try runner.withPreparedInput(
      image: image,
      imageProcessingOptions: imageProcessingOptions,
      expectedMode: .liveStream
    ) { cImage, cOptionsPtr in
      try MpStatus.check(defaultMessage: "Failed to submit live-stream frame for face detection.") {
        MpFaceDetectorDetectAsync(cDetector, cImage, cOptionsPtr, cTimestamp, &$0)
      }
    }
  }
}
