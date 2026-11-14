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
package import third_party_mediapipe_tasks_c_vision_object_detector_object_detector_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
#endif

/// Bounding-box detections and categories produced by `ObjectDetector`.
public struct ObjectDetectorResult: Sendable, Equatable, Hashable {
  /// Detected objects with bounding boxes in unrotated input pixel coordinates.
  public let detections: [Detection]

  /// Frame timestamp in milliseconds (`0` in `.image` mode).
  public let timestampInMilliseconds: Int

  /// Creates an `ObjectDetectorResult` with the given detections and frame timestamp.
  public init(detections: [Detection], timestampInMilliseconds: Int) {
    self.detections = detections
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  /// Creates an `ObjectDetectorResult` copied from a C `MpObjectDetectorResult` struct.
  package init(cResult: MpObjectDetectorResult, timestampInMilliseconds: Int) {
    self.detections = DetectionResult(cResult: cResult).detections
    self.timestampInMilliseconds = timestampInMilliseconds
  }
}
