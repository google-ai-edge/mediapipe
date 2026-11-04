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
package import third_party_mediapipe_tasks_c_components_containers_landmark

/// A 3D point in world space with coordinates measured in meters.
public struct Landmark: Sendable, Equatable, Hashable {
  /// The x coordinate in meters.
  public let x: Float

  /// The y coordinate in meters.
  public let y: Float

  /// The z coordinate (depth) in meters.
  public let z: Float

  /// Optional visibility score indicating whether the landmark is unoccluded and within bounds.
  public let visibility: Float?

  /// Optional presence score indicating whether the landmark is present in the scene.
  public let presence: Float?

  /// Optional semantic name of the landmark.
  public let name: String?

  /// Creates a `Landmark` with the specified 3D coordinates and optional metadata.
  public init(
    x: Float, y: Float, z: Float,
    visibility: Float? = nil, presence: Float? = nil, name: String? = nil
  ) {
    self.x = x
    self.y = y
    self.z = z
    self.visibility = visibility
    self.presence = presence
    self.name = name
  }

  /// Creates a `Landmark` copied from a C `MpLandmark` struct.
  package init(cLandmark: MpLandmark) {
    self.x = cLandmark.x
    self.y = cLandmark.y
    self.z = cLandmark.z
    self.visibility = cLandmark.has_visibility ? cLandmark.visibility : nil
    self.presence = cLandmark.has_presence ? cLandmark.presence : nil
    self.name = cLandmark.name.map { String(cString: $0) }
  }

  /// Returns an array of `Landmark` values copied from a C `MpLandmarks` struct.
  package static func landmarks(from cLandmarks: MpLandmarks) -> [Landmark] {
    Array(cArray: cLandmarks.landmarks, count: cLandmarks.landmarks_count, transform: Self.init)
  }
}

/// A 3D point with `x` and `y` coordinates normalized to `[0.0, 1.0]` by the image dimensions.
public struct NormalizedLandmark: Sendable, Equatable, Hashable {
  /// The normalized x coordinate in `[0.0, 1.0]`.
  public let x: Float

  /// The normalized y coordinate in `[0.0, 1.0]`.
  public let y: Float

  /// The relative z depth coordinate scaled proportionally to `x`.
  public let z: Float

  /// Optional visibility score indicating whether the landmark is unoccluded and within bounds.
  public let visibility: Float?

  /// Optional presence score indicating whether the landmark is present in the scene.
  public let presence: Float?

  /// Optional semantic name of the landmark.
  public let name: String?

  /// Creates a `NormalizedLandmark` with the specified normalized coordinates and optional
  /// metadata.
  public init(
    x: Float, y: Float, z: Float,
    visibility: Float? = nil, presence: Float? = nil, name: String? = nil
  ) {
    self.x = x
    self.y = y
    self.z = z
    self.visibility = visibility
    self.presence = presence
    self.name = name
  }

  /// Creates a `NormalizedLandmark` copied from a C `MpNormalizedLandmark` struct.
  package init(cLandmark: MpNormalizedLandmark) {
    self.x = cLandmark.x
    self.y = cLandmark.y
    self.z = cLandmark.z
    self.visibility = cLandmark.has_visibility ? cLandmark.visibility : nil
    self.presence = cLandmark.has_presence ? cLandmark.presence : nil
    self.name = cLandmark.name.map { String(cString: $0) }
  }

  /// Returns an array of `NormalizedLandmark` values copied from a C `MpNormalizedLandmarks`
  /// struct.
  package static func landmarks(from cLandmarks: MpNormalizedLandmarks) -> [NormalizedLandmark] {
    Array(cArray: cLandmarks.landmarks, count: cLandmarks.landmarks_count, transform: Self.init)
  }
}

/// An undirected or directed edge connecting two landmark indices.
public struct Connection: Sendable, Equatable, Hashable {
  /// The starting landmark index of the connection.
  public let start: Int

  /// The ending landmark index of the connection.
  public let end: Int

  /// Creates a `Connection` between the `start` and `end` landmark indices.
  public init(start: Int, end: Int) {
    self.start = start
    self.end = end
  }
}

/// Region of interest (ROI) used to guide interactive segmentation inference.
public struct RegionOfInterest: Sendable, Equatable, Hashable {
  /// Mutually exclusive representations of a segmentation region of interest.
  @frozen public enum Kind: Sendable, Equatable, Hashable {
    /// A single normalized point targeting the object to segment.
    case keypoint(NormalizedKeypoint)

    /// A sequence of normalized points forming scribbles over the object to segment.
    case scribbles([NormalizedKeypoint])
  }

  /// The underlying representation of this region of interest.
  public let kind: Kind

  /// The single normalized point if initialized with `.keypoint`, or `nil` otherwise.
  public var keypoint: NormalizedKeypoint? {
    guard case .keypoint(let point) = kind else { return nil }
    return point
  }

  /// The scribble keypoints if initialized with `.scribbles`, or `nil` otherwise.
  public var scribbles: [NormalizedKeypoint]? {
    guard case .scribbles(let points) = kind else { return nil }
    return points
  }

  /// Creates a `RegionOfInterest` targeting a single normalized keypoint.
  public init(keypoint: NormalizedKeypoint) {
    self.kind = .keypoint(keypoint)
  }

  /// Creates a `RegionOfInterest` from an array of normalized scribble keypoints.
  public init(scribbles: [NormalizedKeypoint]) {
    self.kind = .scribbles(scribbles)
  }
}
