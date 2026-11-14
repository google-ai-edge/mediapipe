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

import MediaPipeTasksFaceLandmarksConnections
import XCTest

final class FaceLandmarksConnectionsSwiftTests: XCTestCase {
  private static let faceMeshLandmarkCount = 478

  private func edges(_ connections: [Connection]) -> [[Int]] {
    connections.map { [$0.start, $0.end] }
  }

  func testFeatureConnectionCountsMatchFaceMeshTopology() {
    XCTAssertEqual(FaceLandmarksConnections.lips.count, 40)
    XCTAssertEqual(FaceLandmarksConnections.leftEye.count, 16)
    XCTAssertEqual(FaceLandmarksConnections.leftEyebrow.count, 8)
    XCTAssertEqual(FaceLandmarksConnections.leftIris.count, 4)
    XCTAssertEqual(FaceLandmarksConnections.rightEye.count, 16)
    XCTAssertEqual(FaceLandmarksConnections.rightEyebrow.count, 8)
    XCTAssertEqual(FaceLandmarksConnections.rightIris.count, 4)
    XCTAssertEqual(FaceLandmarksConnections.faceOval.count, 36)
    XCTAssertEqual(FaceLandmarksConnections.tesselation.count, 2556)
  }

  func testIrisConnectionsFormDistinctClosedLoops() {
    XCTAssertEqual(
      edges(FaceLandmarksConnections.leftIris),
      [[474, 475], [475, 476], [476, 477], [477, 474]])
    // The legacy Objective-C `rightIrisConnections` returned the left iris by mistake.
    XCTAssertEqual(
      edges(FaceLandmarksConnections.rightIris),
      [[469, 470], [470, 471], [471, 472], [472, 469]])
  }

  func testCompositeConnectionsConcatenateFeatureGroupsInOrder() {
    let features: [[Connection]] = [
      FaceLandmarksConnections.lips, FaceLandmarksConnections.leftEye,
      FaceLandmarksConnections.leftEyebrow, FaceLandmarksConnections.leftIris,
      FaceLandmarksConnections.rightEye, FaceLandmarksConnections.rightEyebrow,
      FaceLandmarksConnections.rightIris, FaceLandmarksConnections.faceOval,
    ]
    XCTAssertEqual(FaceLandmarksConnections.contours, features.flatMap { $0 })
    XCTAssertEqual(
      FaceLandmarksConnections.faceConnections,
      FaceLandmarksConnections.lips + FaceLandmarksConnections.contours
        + FaceLandmarksConnections.tesselation)
    XCTAssertEqual(FaceLandmarksConnections.faceConnections.count, 40 + 132 + 2556)
  }

  func testConnectionEndpointsMatchLegacyHeaderAndStayInFaceMeshRange() {
    XCTAssertEqual(FaceLandmarksConnections.lips.first, Connection(start: 61, end: 146))
    XCTAssertEqual(FaceLandmarksConnections.faceOval.first, Connection(start: 10, end: 338))
    XCTAssertEqual(FaceLandmarksConnections.faceOval.last, Connection(start: 109, end: 10))
    XCTAssertEqual(FaceLandmarksConnections.tesselation.first, Connection(start: 127, end: 34))
    XCTAssertEqual(FaceLandmarksConnections.tesselation.last, Connection(start: 255, end: 339))
    let landmarkRange = 0..<Self.faceMeshLandmarkCount
    XCTAssertTrue(
      FaceLandmarksConnections.faceConnections.allSatisfy {
        landmarkRange.contains($0.start) && landmarkRange.contains($0.end)
      })
  }
}
