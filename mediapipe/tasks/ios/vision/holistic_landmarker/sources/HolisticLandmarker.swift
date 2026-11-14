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
private import third_party_mediapipe_tasks_c_vision_holistic_landmarker_holistic_landmarker_lib
private import third_party_mediapipe_tasks_c_vision_holistic_landmarker_holistic_landmarker_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Detects face, pose, and hand landmarks in upright (`0°`, full-frame) images, videos, or live
/// streams.
///
/// - Note: `HolisticLandmarker` is not thread-safe; serialize concurrent access (for example,
///   inside an `actor`).
public final class HolisticLandmarker {
  private let cLandmarker: MpHolisticLandmarkerPtr
  private let runner: VisionTaskRunner
  private let liveStreamRegistration: VisionLiveStreamRegistration?

  /// Creates a `HolisticLandmarker` configured with `options`.
  public init(options: HolisticLandmarkerOptions) throws {
    let runner = try VisionTaskRunner(
      baseOptions: options.baseOptions, runningMode: options.runningMode,
      roiAllowed: false, normRectAllowed: false,
      hasLiveStreamHandler: options.liveStreamResultHandler != nil)
    let registered = try options.liveStreamResultHandler.map { handler in
      try VisionLiveStreamRegistration.register(
        queueLabelSuffix: "holisticLandmarker", resultType: MpHolisticLandmarkerResult.self,
        as: MpHolisticLandmarkerResultCallback.self,
        decode: { cResult, _, ts in
          try HolisticLandmarkerResult(cResult: cResult.pointee, timestampInMilliseconds: ts)
        }, handler: handler)
    }
    do {
      self.cLandmarker = try options.withCOptions(resultCallback: registered?.cCallback) { cOpts in
        var ptr: MpHolisticLandmarkerPtr?
        try MpStatus.check(defaultMessage: "Failed to create HolisticLandmarker.") { errorMsg in
          MpHolisticLandmarkerCreate(&cOpts, &ptr, &errorMsg)
        }
        guard let ptr else {
          throw TasksError.internalError("MpHolisticLandmarkerCreate returned a null instance.")
        }
        return ptr
      }
    } catch {
      registered?.registration.unregister()
      throw error
    }
    self.runner = runner
    self.liveStreamRegistration = registered?.registration
  }

  /// Creates a `HolisticLandmarker` in `.image` mode from a local `.task` model bundle path.
  public convenience init(modelPath: String) throws {
    try self.init(
      options: HolisticLandmarkerOptions(baseOptions: BaseOptions(modelAssetPath: modelPath)))
  }

  deinit {
    _ = MpHolisticLandmarkerClose(cLandmarker, nil)
    liveStreamRegistration?.unregister()
  }

  /// Detects holistic landmarks in a standalone `image` (`.image` running mode only).
  public func detect(
    image: MPImage, imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> HolisticLandmarkerResult {
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .image
    ) { cImage, cOpts in
      var cResult = MpHolisticLandmarkerResult()
      defer { MpHolisticLandmarkerCloseResult(&cResult) }
      try MpStatus.check(defaultMessage: "Failed to detect holistic landmarks.") { err in
        MpHolisticLandmarkerDetectImage(cLandmarker, cImage, cOpts, &cResult, &err)
      }
      return try HolisticLandmarkerResult(cResult: cResult, timestampInMilliseconds: 0)
    }
  }

  /// Detects holistic landmarks in `videoFrame` at monotonically increasing
  /// `timestampInMilliseconds`.
  public func detect(
    videoFrame: MPImage, timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> HolisticLandmarkerResult {
    let ts = try runner.cTimestamp(timestampInMilliseconds)
    return try runner.withPreparedInput(
      image: videoFrame, imageProcessingOptions: imageProcessingOptions, expectedMode: .video
    ) { cImage, cOpts in
      var cResult = MpHolisticLandmarkerResult()
      defer { MpHolisticLandmarkerCloseResult(&cResult) }
      try MpStatus.check(defaultMessage: "Failed to detect holistic landmarks in video.") { err in
        MpHolisticLandmarkerDetectForVideo(cLandmarker, cImage, cOpts, ts, &cResult, &err)
      }
      return try HolisticLandmarkerResult(
        cResult: cResult, timestampInMilliseconds: timestampInMilliseconds)
    }
  }

  /// Submits `image` at monotonically increasing `timestampInMilliseconds` in `.liveStream` mode.
  public func detectAsync(
    image: MPImage, timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws {
    let ts = try runner.cTimestamp(timestampInMilliseconds)
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .liveStream
    ) { cImage, cOpts in
      try MpStatus.check(defaultMessage: "Failed to submit HolisticLandmarker frame.") { err in
        MpHolisticLandmarkerDetectAsync(cLandmarker, cImage, cOpts, ts, &err)
      }
    }
  }
}
