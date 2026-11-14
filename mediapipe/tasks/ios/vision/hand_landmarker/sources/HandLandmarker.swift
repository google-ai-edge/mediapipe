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
package import third_party_mediapipe_tasks_c_vision_hand_landmarker_hand_landmarker_lib
package import third_party_mediapipe_tasks_c_vision_hand_landmarker_hand_landmarker_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Detects 2D normalized hand landmarks, 3D world landmarks, and handedness across images, video
/// frames, and live camera streams.
///
/// - Note: `HandLandmarker` is not thread-safe. Synchronize concurrent calls externally across
///   threads.
public final class HandLandmarker {
  private static func edges(_ pairs: (Int, Int)...) -> [Connection] { pairs.map(Connection.init) }

  /// Connections linking landmarks across the palm (`wrist` and finger bases).
  public static let handPalmConnections: [Connection] = edges(
    (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))
  /// Connections linking landmarks along the thumb (`1` through `4`).
  public static let handThumbConnections: [Connection] = edges((1, 2), (2, 3), (3, 4))
  /// Connections linking landmarks along the index finger (`5` through `8`).
  public static let handIndexFingerConnections: [Connection] = edges((5, 6), (6, 7), (7, 8))
  /// Connections linking landmarks along the middle finger (`9` through `12`).
  public static let handMiddleFingerConnections: [Connection] = edges((9, 10), (10, 11), (11, 12))
  /// Connections linking landmarks along the ring finger (`13` through `16`).
  public static let handRingFingerConnections: [Connection] = edges((13, 14), (14, 15), (15, 16))
  /// Connections linking landmarks along the pinky finger (`17` through `20`).
  public static let handPinkyConnections: [Connection] = edges((17, 18), (18, 19), (19, 20))
  /// Complete skeleton topology combining palm and all five finger connection lists.
  public static let handConnections: [Connection] =
    handPalmConnections + handThumbConnections + handIndexFingerConnections
    + handMiddleFingerConnections + handRingFingerConnections + handPinkyConnections

  private let cLandmarker: MpHandLandmarkerPtr
  private let runner: VisionTaskRunner
  private let liveStreamRegistration: VisionLiveStreamRegistration?

  /// Creates a `HandLandmarker` configured with `options`.
  public init(options: HandLandmarkerOptions) throws {
    let runner = try VisionTaskRunner(
      baseOptions: options.baseOptions,
      runningMode: options.runningMode,
      roiAllowed: false,
      normRectAllowed: true,
      hasLiveStreamHandler: options.liveStreamResultHandler != nil
    )
    var registration: VisionLiveStreamRegistration?
    var cCallback: MpHandLandmarkerResultCallback?
    if let handler = options.liveStreamResultHandler {
      let registered = try VisionLiveStreamRegistration.register(
        queueLabelSuffix: "handLandmarker",
        resultType: MpHandLandmarkerResult.self,
        as: MpHandLandmarkerResultCallback.self,
        decode: { ptr, _, ts in
          HandLandmarkerResult(cResult: ptr.pointee, timestampInMilliseconds: ts)
        },
        handler: handler
      )
      registration = registered.registration
      cCallback = registered.cCallback
    }
    var succeeded = false
    defer { if !succeeded { registration?.unregister() } }
    var landmarkerPtr: MpHandLandmarkerPtr?
    try options.withCOptions(cCallback: cCallback) { cOptionsPtr in
      try MpStatus.check(defaultMessage: "Failed to create HandLandmarker.") { errorMsg in
        MpHandLandmarkerCreate(cOptionsPtr, &landmarkerPtr, &errorMsg)
      }
    }
    guard let landmarkerPtr else {
      throw TasksError.internalError("MpHandLandmarkerCreate succeeded without returning a task.")
    }
    succeeded = true
    self.cLandmarker = landmarkerPtr
    self.runner = runner
    self.liveStreamRegistration = registration
  }

  /// Creates a `HandLandmarker` in `.image` mode using the model asset bundle at `modelPath`.
  public convenience init(modelPath: String) throws {
    try self.init(
      options: HandLandmarkerOptions(baseOptions: BaseOptions(modelAssetPath: modelPath)))
  }

  deinit {
    _ = MpHandLandmarkerClose(cLandmarker, nil)
    liveStreamRegistration?.unregister()
  }

  /// Detects hand landmarks in a standalone `image` (`.image` running mode only).
  public func detect(
    image: MPImage, imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> HandLandmarkerResult {
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .image
    ) { cImage, cOpts in
      var cResult = MpHandLandmarkerResult()
      defer { MpHandLandmarkerCloseResult(&cResult) }
      try MpStatus.check(defaultMessage: "Failed to detect hand landmarks in image.") { errorMsg in
        MpHandLandmarkerDetectImage(cLandmarker, cImage, cOpts, &cResult, &errorMsg)
      }
      return HandLandmarkerResult(cResult: cResult, timestampInMilliseconds: 0)
    }
  }

  /// Detects hand landmarks in a sequential `videoFrame` at monotonically increasing
  /// `timestampInMilliseconds` (`.video` running mode only).
  public func detect(
    videoFrame: MPImage, timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> HandLandmarkerResult {
    try runner.withPreparedInput(
      image: videoFrame, imageProcessingOptions: imageProcessingOptions, expectedMode: .video
    ) { cImage, cOpts in
      let cTs = try runner.cTimestamp(timestampInMilliseconds)
      var cResult = MpHandLandmarkerResult()
      defer { MpHandLandmarkerCloseResult(&cResult) }
      try MpStatus.check(defaultMessage: "Failed to detect hand landmarks in video frame.") { err in
        MpHandLandmarkerDetectForVideo(cLandmarker, cImage, cOpts, cTs, &cResult, &err)
      }
      return HandLandmarkerResult(
        cResult: cResult, timestampInMilliseconds: timestampInMilliseconds)
    }
  }

  /// Submits a live camera `image` at monotonically increasing `timestampInMilliseconds` for
  /// asynchronous result delivery (`.liveStream` running mode only).
  public func detectAsync(
    image: MPImage, timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws {
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .liveStream
    ) { cImage, cOpts in
      let cTs = try runner.cTimestamp(timestampInMilliseconds)
      try MpStatus.check(defaultMessage: "Failed to send live-stream frame to HandLandmarker.") {
        MpHandLandmarkerDetectAsync(cLandmarker, cImage, cOpts, cTs, &$0)
      }
    }
  }
}
