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
package import third_party_mediapipe_tasks_c_vision_gesture_recognizer_gesture_recognizer_lib
package import third_party_mediapipe_tasks_c_vision_gesture_recognizer_gesture_recognizer_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Recognizes hand gestures, handedness, and 3D landmarks in images, videos, and live streams.
///
/// - Note: Instances are not thread-safe; serialize concurrent access (for example, in an `actor`).
public final class GestureRecognizer {
  private let cRecognizer: MpGestureRecognizerPtr
  private let runner: VisionTaskRunner
  private let liveStreamRegistration: VisionLiveStreamRegistration?

  /// Initializes a gesture recognizer configured with `options`.
  ///
  /// - Throws: `TasksError` if option validation or underlying MediaPipe graph creation fails.
  public init(options: GestureRecognizerOptions) throws {
    let handler = options.liveStreamResultHandler
    let runner = try VisionTaskRunner(
      baseOptions: options.baseOptions, runningMode: options.runningMode,
      roiAllowed: false, normRectAllowed: true, hasLiveStreamHandler: handler != nil)

    var registration: VisionLiveStreamRegistration?
    var cCallback: MpGestureRecognizerResultCallback?
    if options.runningMode == .liveStream, let handler {
      let registered = try VisionLiveStreamRegistration.register(
        queueLabelSuffix: "gesture_recognizer", resultType: MpGestureRecognizerResult.self,
        as: MpGestureRecognizerResultCallback.self,
        decode: { ptr, _, ts in
          GestureRecognizerResult(cResult: ptr.pointee, timestampInMilliseconds: ts)
        }, handler: handler)
      registration = registered.registration
      cCallback = registered.cCallback
    }

    do {
      self.cRecognizer = try options.withCOptions(cCallback: cCallback) { cOptions in
        var recognizerPtr: MpGestureRecognizerPtr?
        try MpStatus.check(defaultMessage: "Failed to create GestureRecognizer.") { errorMessage in
          MpGestureRecognizerCreate(&cOptions, &recognizerPtr, &errorMessage)
        }
        guard let validRecognizer = recognizerPtr else {
          throw TasksError.internalError("MpGestureRecognizerCreate returned a null pointer.")
        }
        return validRecognizer
      }
      self.runner = runner
      self.liveStreamRegistration = registration
    } catch {
      registration?.unregister()
      throw error
    }
  }

  /// Initializes a recognizer in `.image` mode from a `.task` bundle path with default options.
  public convenience init(modelPath: String) throws {
    try self.init(
      options: GestureRecognizerOptions(baseOptions: BaseOptions(modelAssetPath: modelPath)))
  }

  deinit {
    _ = MpGestureRecognizerClose(cRecognizer, nil)
    liveStreamRegistration?.unregister()
  }

  /// Recognizes hand gestures in a still `image` when running in `.image` mode.
  public func recognize(
    image: MPImage, imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> GestureRecognizerResult {
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .image
    ) { cImage, cOpts in
      var cResult = MpGestureRecognizerResult()
      try MpStatus.check(defaultMessage: "Failed to recognize image.") { errorMessage in
        MpGestureRecognizerRecognizeImage(cRecognizer, cImage, cOpts, &cResult, &errorMessage)
      }
      defer { MpGestureRecognizerCloseResult(&cResult) }
      return GestureRecognizerResult(cResult: cResult, timestampInMilliseconds: 0)
    }
  }

  /// Recognizes hand gestures in `videoFrame` at monotonically increasing `timestampInMilliseconds`
  /// when running in `.video` mode.
  public func recognize(
    videoFrame: MPImage, timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> GestureRecognizerResult {
    let ts = try runner.cTimestamp(timestampInMilliseconds)
    return try runner.withPreparedInput(
      image: videoFrame, imageProcessingOptions: imageProcessingOptions, expectedMode: .video
    ) { cImage, cOpts in
      var cResult = MpGestureRecognizerResult()
      try MpStatus.check(defaultMessage: "Failed to recognize video frame.") { err in
        MpGestureRecognizerRecognizeForVideo(cRecognizer, cImage, cOpts, ts, &cResult, &err)
      }
      defer { MpGestureRecognizerCloseResult(&cResult) }
      return GestureRecognizerResult(
        cResult: cResult, timestampInMilliseconds: timestampInMilliseconds)
    }
  }

  /// Enqueues `image` at monotonically increasing `timestampInMilliseconds` in `.liveStream` mode.
  public func recognizeAsync(
    image: MPImage, timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws {
    let ts = try runner.cTimestamp(timestampInMilliseconds)
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .liveStream
    ) { cImage, cOpts in
      try MpStatus.check(defaultMessage: "Failed to recognize live-stream image.") { err in
        MpGestureRecognizerRecognizeAsync(cRecognizer, cImage, cOpts, ts, &err)
      }
    }
  }
}
