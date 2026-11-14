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
private import third_party_mediapipe_tasks_c_core_base_options
private import third_party_mediapipe_tasks_c_vision_core_image
package import third_party_mediapipe_tasks_c_vision_pose_landmarker_pose_landmarker_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Configuration options for initializing a `PoseLandmarker`.
///
/// Not `Equatable` or `Hashable` because `liveStreamResultHandler` is an escaping closure.
public struct PoseLandmarkerOptions: Sendable {
  /// Asynchronous callback receiving live-stream `PoseLandmarker` results and frame timestamp in ms.
  public typealias LiveStreamResultHandler =
    @Sendable (Result<PoseLandmarkerResult, any Error>, _ timestampInMilliseconds: Int) -> Void

  /// Model asset loading and hardware acceleration settings.
  public var baseOptions: BaseOptions
  /// Execution mode (`.image`, `.video`, or `.liveStream`).
  public var runningMode: VisionRunningMode
  /// Maximum number of poses detected per frame.
  public var numPoses: Int
  /// Minimum confidence score in `[0.0, 1.0]` for pose detection to succeed.
  public var minPoseDetectionConfidence: Float
  /// Minimum confidence score in `[0.0, 1.0]` for pose presence in landmark detection.
  public var minPosePresenceConfidence: Float
  /// Minimum confidence score in `[0.0, 1.0]` for pose tracking across frames.
  public var minTrackingConfidence: Float
  /// Whether the landmarker produces per-pose segmentation masks.
  public var outputSegmentationMasks: Bool
  /// Callback required when `runningMode == .liveStream` and disallowed in `.image`/`.video` modes.
  public var liveStreamResultHandler: LiveStreamResultHandler?

  /// Creates `PoseLandmarkerOptions` defaulting to single-pose `.image` mode without masks.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    runningMode: VisionRunningMode = .image,
    numPoses: Int = 1,
    minPoseDetectionConfidence: Float = 0.5,
    minPosePresenceConfidence: Float = 0.5,
    minTrackingConfidence: Float = 0.5,
    outputSegmentationMasks: Bool = false,
    liveStreamResultHandler: LiveStreamResultHandler? = nil
  ) {
    self.baseOptions = baseOptions
    self.runningMode = runningMode
    self.numPoses = numPoses
    self.minPoseDetectionConfidence = minPoseDetectionConfidence
    self.minPosePresenceConfidence = minPosePresenceConfidence
    self.minTrackingConfidence = minTrackingConfidence
    self.outputSegmentationMasks = outputSegmentationMasks
    self.liveStreamResultHandler = liveStreamResultHandler
  }

  package func withCOptions<Result>(
    cCallback: MpPoseLandmarkerResultCallback?,
    _ body: (inout MpPoseLandmarkerOptions) throws -> Result
  ) throws -> Result {
    guard let cNumPoses = Int32(exactly: numPoses), cNumPoses > 0 else {
      throw TasksError.invalidArgument("numPoses must be a positive 32-bit integer.")
    }
    return try baseOptions.withCOptions { cBaseOptions in
      var cOptions = MpPoseLandmarkerOptions(
        base_options: cBaseOptions,
        running_mode: runningMode.cRunningMode,
        num_poses: cNumPoses,
        min_pose_detection_confidence: minPoseDetectionConfidence,
        min_pose_presence_confidence: minPosePresenceConfidence,
        min_tracking_confidence: minTrackingConfidence,
        output_segmentation_masks: outputSegmentationMasks,
        result_callback: cCallback
      )
      return try body(&cOptions)
    }
  }
}
