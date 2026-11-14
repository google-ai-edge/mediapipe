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
package import third_party_mediapipe_tasks_c_vision_holistic_landmarker_holistic_landmarker_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Configuration options for initializing a MediaPipe `HolisticLandmarker` task.
///
/// Not `Equatable` or `Hashable` because `liveStreamResultHandler` is a closure.
public struct HolisticLandmarkerOptions: Sendable {
  /// Asynchronous callback invoked on a serial background queue for `.liveStream` detection.
  public typealias LiveStreamResultHandler =
    @Sendable (Result<HolisticLandmarkerResult, any Error>, _ timestampInMilliseconds: Int) -> Void

  /// Shared options for loading the model bundle and configuring hardware acceleration.
  public var baseOptions: BaseOptions
  /// Execution mode (`.image`, `.video`, or `.liveStream`) for the landmarker.
  public var runningMode: VisionRunningMode
  /// Minimum confidence score in `[0.0, 1.0]` for face detection to succeed.
  public var minFaceDetectionConfidence: Float
  /// Minimum non-maximum-suppression IoU threshold in `[0.0, 1.0]` for face detection.
  public var minFaceSuppressionThreshold: Float
  /// Minimum face-presence confidence score in `[0.0, 1.0]` for face landmark detection.
  public var minFacePresenceConfidence: Float
  /// Minimum confidence score in `[0.0, 1.0]` for pose detection to succeed.
  public var minPoseDetectionConfidence: Float
  /// Minimum non-maximum-suppression IoU threshold in `[0.0, 1.0]` for pose detection.
  public var minPoseSuppressionThreshold: Float
  /// Minimum pose-presence confidence score in `[0.0, 1.0]` for pose landmark detection.
  public var minPosePresenceConfidence: Float
  /// Minimum hand-presence confidence score in `[0.0, 1.0]` for hand landmark detection.
  public var minHandLandmarksConfidence: Float
  /// Whether to output 3D face animation blendshape classifications.
  public var outputFaceBlendshapes: Bool
  /// Whether to output `Float32` pose segmentation masks.
  public var outputPoseSegmentationMasks: Bool
  /// Handler invoked asynchronously with detection results when `runningMode == .liveStream`.
  public var liveStreamResultHandler: LiveStreamResultHandler?

  /// Creates `HolisticLandmarkerOptions` with default or custom detection thresholds and outputs.
  public init(
    baseOptions: BaseOptions = BaseOptions(), runningMode: VisionRunningMode = .image,
    minFaceDetectionConfidence: Float = 0.5, minFaceSuppressionThreshold: Float = 0.3,
    minFacePresenceConfidence: Float = 0.5, minPoseDetectionConfidence: Float = 0.5,
    minPoseSuppressionThreshold: Float = 0.3, minPosePresenceConfidence: Float = 0.5,
    minHandLandmarksConfidence: Float = 0.5, outputFaceBlendshapes: Bool = false,
    outputPoseSegmentationMasks: Bool = false,
    liveStreamResultHandler: LiveStreamResultHandler? = nil
  ) {
    self.baseOptions = baseOptions
    self.runningMode = runningMode
    self.minFaceDetectionConfidence = minFaceDetectionConfidence
    self.minFaceSuppressionThreshold = minFaceSuppressionThreshold
    self.minFacePresenceConfidence = minFacePresenceConfidence
    self.minPoseDetectionConfidence = minPoseDetectionConfidence
    self.minPoseSuppressionThreshold = minPoseSuppressionThreshold
    self.minPosePresenceConfidence = minPosePresenceConfidence
    self.minHandLandmarksConfidence = minHandLandmarksConfidence
    self.outputFaceBlendshapes = outputFaceBlendshapes
    self.outputPoseSegmentationMasks = outputPoseSegmentationMasks
    self.liveStreamResultHandler = liveStreamResultHandler
  }

  package func withCOptions<Result>(
    resultCallback: MpHolisticLandmarkerResultCallback? = nil,
    _ body: (inout MpHolisticLandmarkerOptions) throws -> Result
  ) throws -> Result {
    try baseOptions.withCOptions { cBaseOptions in
      var cOptions = MpHolisticLandmarkerOptions()
      cOptions.base_options = cBaseOptions
      cOptions.running_mode = runningMode.cRunningMode
      cOptions.min_face_detection_confidence = minFaceDetectionConfidence
      cOptions.min_face_suppression_threshold = minFaceSuppressionThreshold
      cOptions.min_face_presence_confidence = minFacePresenceConfidence
      cOptions.min_pose_detection_confidence = minPoseDetectionConfidence
      cOptions.min_pose_suppression_threshold = minPoseSuppressionThreshold
      cOptions.min_pose_presence_confidence = minPosePresenceConfidence
      cOptions.min_hand_landmarks_confidence = minHandLandmarksConfidence
      cOptions.output_face_blendshapes = outputFaceBlendshapes
      cOptions.output_pose_segmentation_masks = outputPoseSegmentationMasks
      cOptions.result_callback = resultCallback
      return try body(&cOptions)
    }
  }
}
