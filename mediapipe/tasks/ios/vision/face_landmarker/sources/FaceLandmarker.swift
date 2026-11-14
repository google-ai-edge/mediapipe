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
package import third_party_mediapipe_tasks_c_vision_face_landmarker_face_landmarker_lib
package import third_party_mediapipe_tasks_c_vision_face_landmarker_face_landmarker_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksFaceLandmarksConnections
  @_exported import MediaPipeTasksVisionCore
#endif

/// Detects 478-point face mesh landmarks, optional blendshapes, and optional facial transformation
/// matrixes in images, video frames, or live camera streams. Region-of-interest is not supported.
///
/// - Note: This class is not thread-safe. Concurrent callers should serialize access (for example,
///   by isolating an instance inside a Swift `actor`).
public final class FaceLandmarker {
  private let cLandmarker: MpFaceLandmarkerPtr
  private let runner: VisionTaskRunner
  private let liveStreamRegistration: VisionLiveStreamRegistration?

  /// Lips connections of the face mesh.
  public static var lipsConnections: [Connection] { FaceLandmarksConnections.lips }
  /// Left eye connections of the face mesh.
  public static var leftEyeConnections: [Connection] { FaceLandmarksConnections.leftEye }
  /// Left eyebrow connections of the face mesh.
  public static var leftEyebrowConnections: [Connection] { FaceLandmarksConnections.leftEyebrow }
  /// Left iris connections of the face mesh.
  public static var leftIrisConnections: [Connection] { FaceLandmarksConnections.leftIris }
  /// Right eye connections of the face mesh.
  public static var rightEyeConnections: [Connection] { FaceLandmarksConnections.rightEye }
  /// Right eyebrow connections of the face mesh.
  public static var rightEyebrowConnections: [Connection] { FaceLandmarksConnections.rightEyebrow }
  /// Right iris connections of the face mesh.
  public static var rightIrisConnections: [Connection] { FaceLandmarksConnections.rightIris }
  /// Face oval connections of the face mesh.
  public static var faceOvalConnections: [Connection] { FaceLandmarksConnections.faceOval }
  /// All feature contour connections of the face mesh.
  public static var contoursConnections: [Connection] { FaceLandmarksConnections.contours }
  /// Tesselation connections of the face mesh.
  public static var tesselationConnections: [Connection] { FaceLandmarksConnections.tesselation }
  /// Lips, contours, and tesselation connections of the face mesh.
  public static var faceConnections: [Connection] { FaceLandmarksConnections.faceConnections }

  /// Creates a face landmarker and initializes the underlying MediaPipe task graph.
  ///
  /// - Throws: `TasksError` if `options` are inconsistent or graph initialization fails.
  public init(options: FaceLandmarkerOptions) throws {
    let runner = try VisionTaskRunner(
      baseOptions: options.baseOptions, runningMode: options.runningMode, roiAllowed: false,
      normRectAllowed: true, hasLiveStreamHandler: options.liveStreamResultHandler != nil)
    let registered = try options.liveStreamResultHandler.map { handler in
      try VisionLiveStreamRegistration.register(
        queueLabelSuffix: "faceLandmarker", resultType: MpFaceLandmarkerResult.self,
        as: MpFaceLandmarkerResultCallback.self,
        decode: { FaceLandmarkerResult(cResult: $0.pointee, timestampInMilliseconds: $2) },
        handler: handler)
    }
    do {
      self.cLandmarker = try options.withCOptions(resultCallback: registered?.cCallback) {
        cOptions in
        var landmarkerPtr: MpFaceLandmarkerPtr?
        try MpStatus.check(defaultMessage: "Failed to create FaceLandmarker.") { errorMessage in
          MpFaceLandmarkerCreate(&cOptions, &landmarkerPtr, &errorMessage)
        }
        guard let landmarkerPtr else {
          throw TasksError.internalError("MpFaceLandmarkerCreate returned a null pointer.")
        }
        return landmarkerPtr
      }
    } catch {
      registered?.registration.unregister()
      throw error
    }
    self.runner = runner
    self.liveStreamRegistration = registered?.registration
  }

  /// Creates a face landmarker in `.image` mode from a local `.task` model file path.
  public convenience init(modelPath: String) throws {
    try self.init(
      options: FaceLandmarkerOptions(baseOptions: BaseOptions(modelAssetPath: modelPath)))
  }

  deinit {
    _ = MpFaceLandmarkerClose(cLandmarker, nil)
    liveStreamRegistration?.unregister()
  }

  /// Detects face landmarks in a single image. Requires `.image` running mode.
  public func detect(
    image: MPImage,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> FaceLandmarkerResult {
    try runDetection(image, imageProcessingOptions, mode: .image, timestamp: 0) {
      MpFaceLandmarkerDetectImage(cLandmarker, $0, $1, $2, &$3)
    }
  }

  /// Detects face landmarks in a decoded video frame. Requires `.video` running mode and
  /// monotonically increasing `timestampInMilliseconds`.
  public func detect(
    videoFrame: MPImage,
    timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> FaceLandmarkerResult {
    let cTimestamp = try runner.cTimestamp(timestampInMilliseconds)
    return try runDetection(
      videoFrame, imageProcessingOptions, mode: .video, timestamp: timestampInMilliseconds
    ) { MpFaceLandmarkerDetectForVideo(cLandmarker, $0, $1, cTimestamp, $2, &$3) }
  }

  /// Sends a live-stream frame for asynchronous detection, delivering the result to
  /// `FaceLandmarkerOptions.liveStreamResultHandler`. Requires `.liveStream` running mode and
  /// monotonically increasing `timestampInMilliseconds`.
  public func detectAsync(
    image: MPImage,
    timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws {
    let cTimestamp = try runner.cTimestamp(timestampInMilliseconds)
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .liveStream
    ) { cImage, cOptions in
      try MpStatus.check(defaultMessage: "Failed to send frame to FaceLandmarker.") {
        errorMessage in
        MpFaceLandmarkerDetectAsync(cLandmarker, cImage, cOptions, cTimestamp, &errorMessage)
      }
    }
  }

  private func runDetection(
    _ image: MPImage, _ imageProcessingOptions: ImageProcessingOptions?,
    mode: VisionRunningMode, timestamp: Int,
    _ call: (
      MpImagePtr, UnsafePointer<MpImageProcessingOptions>,
      UnsafeMutablePointer<MpFaceLandmarkerResult>, inout UnsafeMutablePointer<CChar>?
    ) -> MpStatus
  ) throws -> FaceLandmarkerResult {
    var cResult = MpFaceLandmarkerResult()
    defer { MpFaceLandmarkerCloseResult(&cResult) }
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: mode
    ) { cImage, cOptions in
      try MpStatus.check(defaultMessage: "Failed to detect face landmarks.") { errorMessage in
        call(cImage, cOptions, &cResult, &errorMessage)
      }
    }
    return FaceLandmarkerResult(cResult: cResult, timestampInMilliseconds: timestamp)
  }
}
