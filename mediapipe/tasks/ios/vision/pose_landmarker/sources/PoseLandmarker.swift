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
private import third_party_mediapipe_tasks_c_vision_pose_landmarker_pose_landmarker_lib
private import third_party_mediapipe_tasks_c_vision_pose_landmarker_pose_landmarker_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Detects 33-point human pose landmarks, world landmarks, and optional segmentation masks across
/// still images, video frames, and live camera streams.
///
/// - Note: `PoseLandmarker` is not thread-safe; synchronize concurrent access externally.
public final class PoseLandmarker {
  /// Standard 33-landmark MediaPipe BlazePose skeleton edge connections.
  public static let poseLandmarks: [Connection] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18),
    (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27),
    (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
  ].map { Connection(start: $0.0, end: $0.1) }

  private let cLandmarker: MpPoseLandmarkerPtr
  private let runner: VisionTaskRunner
  private let outputSegmentationMasks: Bool
  private let liveStreamRegistration: VisionLiveStreamRegistration?

  /// Creates a `PoseLandmarker` configured with `options`.
  public init(options: PoseLandmarkerOptions) throws {
    let handler = options.liveStreamResultHandler
    let outputMasks = options.outputSegmentationMasks
    let runner = try VisionTaskRunner(
      baseOptions: options.baseOptions, runningMode: options.runningMode,
      roiAllowed: false, normRectAllowed: true, hasLiveStreamHandler: handler != nil
    )
    var registration: VisionLiveStreamRegistration?
    var cCallback: MpPoseLandmarkerResultCallback?
    if let handler {
      let registered = try VisionLiveStreamRegistration.register(
        queueLabelSuffix: "poseLandmarker",
        resultType: MpPoseLandmarkerResult.self,
        as: MpPoseLandmarkerResultCallback.self,
        decode: { cResultPtr, _, timestampMs in
          try PoseLandmarkerResult(
            cResult: cResultPtr.pointee, outputSegmentationMasks: outputMasks,
            timestampInMilliseconds: timestampMs)
        },
        handler: handler
      )
      registration = registered.registration
      cCallback = registered.cCallback
    }

    do {
      var landmarkerPtr: MpPoseLandmarkerPtr?
      try options.withCOptions(cCallback: cCallback) { cOptions in
        try MpStatus.check(defaultMessage: "Failed to create PoseLandmarker.") { errorMsg in
          MpPoseLandmarkerCreate(&cOptions, &landmarkerPtr, &errorMsg)
        }
      }
      guard let landmarkerPtr else {
        throw TasksError.internalError("MpPoseLandmarkerCreate returned a null pointer.")
      }
      self.cLandmarker = landmarkerPtr
      self.runner = runner
      self.outputSegmentationMasks = outputMasks
      self.liveStreamRegistration = registration
    } catch {
      registration?.unregister()
      throw error
    }
  }

  /// Creates a `PoseLandmarker` in `.image` mode using the model bundle at `modelPath`.
  public convenience init(modelPath: String) throws {
    try self.init(
      options: PoseLandmarkerOptions(baseOptions: BaseOptions(modelAssetPath: modelPath)))
  }

  deinit {
    _ = MpPoseLandmarkerClose(cLandmarker, nil)
    liveStreamRegistration?.unregister()
  }

  /// Detects pose landmarks in a still `image` (`.image` running mode only).
  public func detect(
    image: MPImage, imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> PoseLandmarkerResult {
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .image
    ) { cImage, cOpts in
      var cResult = MpPoseLandmarkerResult()
      try MpStatus.check(defaultMessage: "Failed to detect pose landmarks in image.") { err in
        MpPoseLandmarkerDetectImage(cLandmarker, cImage, cOpts, &cResult, &err)
      }
      defer { MpPoseLandmarkerCloseResult(&cResult) }
      return try PoseLandmarkerResult(
        cResult: cResult, outputSegmentationMasks: outputSegmentationMasks,
        timestampInMilliseconds: 0)
    }
  }

  /// Detects pose landmarks in `videoFrame` at monotonically increasing `timestampInMilliseconds`.
  public func detect(
    videoFrame: MPImage, timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> PoseLandmarkerResult {
    try runner.withPreparedInput(
      image: videoFrame, imageProcessingOptions: imageProcessingOptions, expectedMode: .video
    ) { cImage, cOpts in
      let cTimestamp = try runner.cTimestamp(timestampInMilliseconds)
      var cResult = MpPoseLandmarkerResult()
      try MpStatus.check(defaultMessage: "Failed to detect pose landmarks in video frame.") { err in
        MpPoseLandmarkerDetectForVideo(cLandmarker, cImage, cOpts, cTimestamp, &cResult, &err)
      }
      defer { MpPoseLandmarkerCloseResult(&cResult) }
      return try PoseLandmarkerResult(
        cResult: cResult, outputSegmentationMasks: outputSegmentationMasks,
        timestampInMilliseconds: timestampInMilliseconds)
    }
  }

  /// Enqueues `image` at monotonically increasing `timestampInMilliseconds` in `.liveStream` mode.
  public func detectAsync(
    image: MPImage, timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws {
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .liveStream
    ) { cImage, cOpts in
      let cTimestamp = try runner.cTimestamp(timestampInMilliseconds)
      try MpStatus.check(defaultMessage: "Failed to detect pose landmarks in live stream.") { err in
        MpPoseLandmarkerDetectAsync(cLandmarker, cImage, cOpts, cTimestamp, &err)
      }
    }
  }
}
