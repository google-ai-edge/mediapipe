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
package import third_party_mediapipe_tasks_c_vision_core_image
package import third_party_mediapipe_tasks_c_vision_face_landmarker_face_landmarker_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Options for configuring a `FaceLandmarker` task (not `Equatable` because
/// `liveStreamResultHandler` is a closure).
public struct FaceLandmarkerOptions: Sendable {
  /// Receives `.liveStream` results (or errors) and input timestamps on a private serial queue.
  public typealias LiveStreamResultHandler =
    @Sendable (Result<FaceLandmarkerResult, any Error>, _ timestampInMilliseconds: Int) -> Void

  /// Base options for model loading and hardware acceleration delegation.
  public var baseOptions: BaseOptions
  /// Execution mode of the landmarker. `.liveStream` requires `liveStreamResultHandler`.
  public var runningMode: VisionRunningMode
  /// Maximum number of faces to detect.
  public var numFaces: Int
  /// Minimum confidence score for face detection to be considered successful.
  public var minFaceDetectionConfidence: Float
  /// Minimum confidence score of the face presence score in face landmark detection.
  public var minFacePresenceConfidence: Float
  /// Minimum confidence score for face tracking to be considered successful.
  public var minTrackingConfidence: Float
  /// Whether to output face blendshape scores for rendering 3D face models.
  public var outputFaceBlendshapes: Bool
  /// Whether to output 4x4 facial transformation matrixes from the canonical face model.
  public var outputFacialTransformationMatrixes: Bool
  /// Handler for `.liveStream` results. Must be `nil` in `.image` and `.video` modes.
  public var liveStreamResultHandler: LiveStreamResultHandler?

  /// Creates options for single-image detection of one face with `0.5` confidence thresholds.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    runningMode: VisionRunningMode = .image,
    numFaces: Int = 1,
    minFaceDetectionConfidence: Float = 0.5,
    minFacePresenceConfidence: Float = 0.5,
    minTrackingConfidence: Float = 0.5,
    outputFaceBlendshapes: Bool = false,
    outputFacialTransformationMatrixes: Bool = false,
    liveStreamResultHandler: LiveStreamResultHandler? = nil
  ) {
    self.baseOptions = baseOptions
    self.runningMode = runningMode
    self.numFaces = numFaces
    self.minFaceDetectionConfidence = minFaceDetectionConfidence
    self.minFacePresenceConfidence = minFacePresenceConfidence
    self.minTrackingConfidence = minTrackingConfidence
    self.outputFaceBlendshapes = outputFaceBlendshapes
    self.outputFacialTransformationMatrixes = outputFacialTransformationMatrixes
    self.liveStreamResultHandler = liveStreamResultHandler
  }

  /// Invokes `body` with a C `MpFaceLandmarkerOptions` struct whose pointer fields remain valid for
  /// the duration of the closure call.
  package func withCOptions<Result>(
    resultCallback: MpFaceLandmarkerResultCallback?,
    _ body: (inout MpFaceLandmarkerOptions) throws -> Result
  ) throws -> Result {
    guard numFaces > 0, let cNumFaces = Int32(exactly: numFaces) else {
      throw TasksError.invalidArgument("numFaces (\(numFaces)) must be a positive Int32.")
    }
    return try baseOptions.withCOptions { cBaseOptions in
      var cOptions = MpFaceLandmarkerOptions()
      cOptions.base_options = cBaseOptions
      cOptions.running_mode = runningMode.cRunningMode
      cOptions.num_faces = cNumFaces
      cOptions.min_face_detection_confidence = minFaceDetectionConfidence
      cOptions.min_face_presence_confidence = minFacePresenceConfidence
      cOptions.min_tracking_confidence = minTrackingConfidence
      cOptions.output_face_blendshapes = outputFaceBlendshapes
      cOptions.output_facial_transformation_matrixes = outputFacialTransformationMatrixes
      cOptions.result_callback = resultCallback
      return try body(&cOptions)
    }
  }
}
