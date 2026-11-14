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

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Configuration options for the MediaPipe `HandLandmarker` task.
public struct HandLandmarkerOptions: Sendable {
  /// Asynchronous callback invoked on a serial queue with live-stream hand landmarking results.
  public typealias LiveStreamResultHandler =
    @Sendable (Result<HandLandmarkerResult, any Error>, _ timestampInMilliseconds: Int) -> Void

  /// Model asset and hardware delegate configuration.
  public var baseOptions: BaseOptions
  /// Execution mode (`image`, `video`, or `liveStream`).
  public var runningMode: VisionRunningMode
  /// Maximum number of hands to detect per frame (must be positive).
  public var numHands: Int
  /// Minimum confidence score for palm detection to succeed.
  public var minHandDetectionConfidence: Float
  /// Minimum hand-presence confidence score during landmark regression.
  public var minHandPresenceConfidence: Float
  /// Minimum landmark-tracking confidence before re-triggering palm detection on the next frame.
  public var minTrackingConfidence: Float
  /// Asynchronous delivery closure required when `runningMode` is `.liveStream`.
  public var liveStreamResultHandler: LiveStreamResultHandler?

  /// Creates `HandLandmarkerOptions` with the specified configuration.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    runningMode: VisionRunningMode = .image,
    numHands: Int = 1,
    minHandDetectionConfidence: Float = 0.5,
    minHandPresenceConfidence: Float = 0.5,
    minTrackingConfidence: Float = 0.5,
    liveStreamResultHandler: LiveStreamResultHandler? = nil
  ) {
    self.baseOptions = baseOptions
    self.runningMode = runningMode
    self.numHands = numHands
    self.minHandDetectionConfidence = minHandDetectionConfidence
    self.minHandPresenceConfidence = minHandPresenceConfidence
    self.minTrackingConfidence = minTrackingConfidence
    self.liveStreamResultHandler = liveStreamResultHandler
  }

  /// Invokes `body` with a mutable C `MpHandLandmarkerOptions` pointer whose nested C buffers
  /// remain valid for the duration of the closure call.
  package func withCOptions<Result>(
    cCallback: MpHandLandmarkerResultCallback?,
    _ body: (UnsafeMutablePointer<MpHandLandmarkerOptions>) throws -> Result
  ) throws -> Result {
    guard numHands > 0, let cNumHands = Int32(exactly: numHands) else {
      throw TasksError.invalidArgument("numHands (\(numHands)) must be a positive Int32.")
    }
    return try baseOptions.withCOptions { cBaseOptions in
      var cOptions = MpHandLandmarkerOptions()
      cOptions.base_options = cBaseOptions
      cOptions.running_mode = runningMode.cRunningMode
      cOptions.num_hands = cNumHands
      cOptions.min_hand_detection_confidence = minHandDetectionConfidence
      cOptions.min_hand_presence_confidence = minHandPresenceConfidence
      cOptions.min_tracking_confidence = minTrackingConfidence
      cOptions.result_callback = cCallback
      return try withUnsafeMutablePointer(to: &cOptions, body)
    }
  }
}
