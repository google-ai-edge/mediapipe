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
package import third_party_mediapipe_tasks_c_components_containers_keypoint
package import third_party_mediapipe_tasks_c_components_containers_rect

#if canImport(CoreGraphics)
  import CoreGraphics
#endif

/// A 2D keypoint with coordinates normalized to `[0.0, 1.0]` by the image width and height.
public struct NormalizedKeypoint: Sendable, Equatable {
  /// Normalized `(x, y)` coordinates of the keypoint.
  public let location: CGPoint

  /// Optional semantic label of the keypoint.
  public let label: String?

  /// Optional confidence score of the keypoint.
  public let score: Float?

  /// Creates a `NormalizedKeypoint` with the given normalized location, optional label, and
  /// optional confidence score.
  public init(location: CGPoint, label: String? = nil, score: Float? = nil) {
    self.location = location
    self.label = label
    self.score = score
  }

  /// Creates a `NormalizedKeypoint` copied from a C `MpNormalizedKeypoint` struct.
  package init(cKeypoint: MpNormalizedKeypoint) {
    self.location = CGPoint(x: Double(cKeypoint.x), y: Double(cKeypoint.y))
    self.label = cKeypoint.label.map { String(cString: $0) }
    self.score = cKeypoint.has_score ? cKeypoint.score : nil
  }
}

extension NormalizedKeypoint: Hashable {
  public func hash(into hasher: inout Hasher) {
    hasher.combine(location.x)
    hasher.combine(location.y)
    hasher.combine(label)
    hasher.combine(score)
  }
}

/// A single detected object with predicted categories, bounding box, and optional keypoints.
public struct Detection: Sendable, Equatable {
  /// Predicted categories for the detected object.
  public let categories: [ResultCategory]

  /// Bounding box in image pixel coordinates.
  public let boundingBox: CGRect

  /// Optional normalized keypoints associated with the detection.
  public let keypoints: [NormalizedKeypoint]?

  /// Creates a `Detection` with the given categories, bounding box, and optional keypoints.
  public init(
    categories: [ResultCategory], boundingBox: CGRect, keypoints: [NormalizedKeypoint]? = nil
  ) {
    self.categories = categories
    self.boundingBox = boundingBox
    self.keypoints = keypoints
  }

  /// Creates a `Detection` copied from a C `MpDetection` struct.
  package init(cDetection: MpDetection) {
    self.categories = Array(
      cArray: cDetection.categories, count: cDetection.categories_count,
      transform: ResultCategory.init
    )

    let rect = cDetection.bounding_box
    self.boundingBox = CGRect(
      x: Int(rect.left), y: Int(rect.top),
      width: Int(rect.right) - Int(rect.left),
      height: Int(rect.bottom) - Int(rect.top)
    )

    if cDetection.keypoints != nil, cDetection.keypoints_count > 0 {
      self.keypoints = Array(
        cArray: cDetection.keypoints, count: cDetection.keypoints_count,
        transform: NormalizedKeypoint.init
      )
    } else {
      self.keypoints = nil
    }
  }
}

extension Detection: Hashable {
  public func hash(into hasher: inout Hasher) {
    hasher.combine(categories)
    if boundingBox.isNull {
      hasher.combine(true)
    } else {
      let box = boundingBox.standardized
      hasher.combine(false)
      hasher.combine(box.origin.x)
      hasher.combine(box.origin.y)
      hasher.combine(box.size.width)
      hasher.combine(box.size.height)
    }
    hasher.combine(keypoints)
  }
}

/// Detection results produced by a detection model.
public struct DetectionResult: Sendable, Equatable, Hashable {
  /// All detections found in the input region.
  public let detections: [Detection]

  /// Creates a `DetectionResult` with the given detections.
  public init(detections: [Detection]) {
    self.detections = detections
  }

  /// Creates a `DetectionResult` copied from a C `MpDetectionResult` struct.
  package init(cResult: MpDetectionResult) {
    self.detections = Array(
      cArray: cResult.detections, count: cResult.detections_count, transform: Detection.init
    )
  }
}
