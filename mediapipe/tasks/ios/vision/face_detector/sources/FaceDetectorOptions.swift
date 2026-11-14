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
package import third_party_mediapipe_tasks_c_vision_face_detector_face_detector_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Configuration options for initializing a MediaPipe `FaceDetector` task.
///
/// Does not conform to `Equatable` or `Hashable` because `liveStreamResultHandler` is a closure.
public struct FaceDetectorOptions: Sendable {
  /// Asynchronous callback invoked on a serial background queue for `.liveStream` face detection.
  public typealias LiveStreamResultHandler =
    @Sendable (Result<FaceDetectorResult, any Error>, _ timestampInMilliseconds: Int) -> Void

  /// Shared options for loading the model asset and configuring the hardware delegate.
  public var baseOptions: BaseOptions

  /// Execution mode (`.image`, `.video`, or `.liveStream`) for the detector.
  public var runningMode: VisionRunningMode

  /// Minimum confidence score in `[0.0, 1.0]` required for a face detection to be retained.
  public var minDetectionConfidence: Float

  /// Minimum non-maximum-suppression IoU threshold in `[0.0, 1.0]` for overlapping detections.
  public var minSuppressionThreshold: Float

  /// Handler invoked asynchronously with detection results when `runningMode == .liveStream`.
  public var liveStreamResultHandler: LiveStreamResultHandler?

  /// Creates `FaceDetectorOptions` with default or custom detection thresholds and running mode.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    runningMode: VisionRunningMode = .image,
    minDetectionConfidence: Float = 0.5,
    minSuppressionThreshold: Float = 0.3,
    liveStreamResultHandler: LiveStreamResultHandler? = nil
  ) {
    self.baseOptions = baseOptions
    self.runningMode = runningMode
    self.minDetectionConfidence = minDetectionConfidence
    self.minSuppressionThreshold = minSuppressionThreshold
    self.liveStreamResultHandler = liveStreamResultHandler
  }

  /// Invokes `body` with a transient C `MpFaceDetectorOptions` struct whose underlying
  /// `BaseOptions` pointers remain valid for the duration of the closure call.
  package func withCOptions<Result>(
    resultCallback: MpFaceDetectorResultCallback? = nil,
    _ body: (inout MpFaceDetectorOptions) throws -> Result
  ) throws -> Result {
    try baseOptions.withCOptions { cBaseOptions in
      var cOptions = MpFaceDetectorOptions()
      cOptions.base_options = cBaseOptions
      cOptions.running_mode = runningMode.cRunningMode
      cOptions.min_detection_confidence = minDetectionConfidence
      cOptions.min_suppression_threshold = minSuppressionThreshold
      cOptions.result_callback = resultCallback
      return try body(&cOptions)
    }
  }
}
