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
package import third_party_mediapipe_tasks_c_components_containers_detection_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Detection results produced by `FaceDetector` for a single image or frame.
public struct FaceDetectorResult: Sendable, Equatable, Hashable {
  /// Detected faces in the unrotated input image coordinate space (`[0, width) x [0, height)`).
  public let detections: [Detection]

  /// Timestamp associated with this detection result, in milliseconds.
  public let timestampInMilliseconds: Int

  /// Creates a `FaceDetectorResult` with explicit face detections and frame timestamp.
  public init(detections: [Detection], timestampInMilliseconds: Int = 0) {
    self.detections = detections
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  /// Copies a C `MpDetectionResult` into an owned Swift `FaceDetectorResult`.
  package init(cResult: MpDetectionResult, timestampInMilliseconds: Int = 0) {
    self.init(
      detections: DetectionResult(cResult: cResult).detections,
      timestampInMilliseconds: timestampInMilliseconds
    )
  }
}
