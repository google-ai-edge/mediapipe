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
import MediaPipeTasksFaceLandmarker
import UIKit
import XCTest
import third_party_mediapipe_tasks_c_components_containers_matrix
import third_party_mediapipe_tasks_c_vision_face_landmarker_face_landmarker_lib

final class FaceLandmarkerSwiftTests: XCTestCase {
  private enum Fixture {
    static let modelName = "face_landmarker_v2_with_blendshapes"
    static let tolerance: Float = 0.05
  }

  private func path(_ name: String, _ ext: String) throws -> String {
    try XCTUnwrap(Bundle(for: Self.self).path(forResource: name, ofType: ext), "\(name).\(ext)")
  }

  private func image(_ name: String) throws -> MPImage {
    try MPImage(uiImage: XCTUnwrap(UIImage(contentsOfFile: path(name, "jpg"))))
  }

  private func options(_ mode: VisionRunningMode = .image) throws -> FaceLandmarkerOptions {
    FaceLandmarkerOptions(
      baseOptions: BaseOptions(modelAssetPath: try path(Fixture.modelName, "task")),
      runningMode: mode, outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true)
  }

  private func assertPortraitResult(
    _ result: FaceLandmarkerResult, firstLandmark: (Float, Float), matrixColumn0: [Float],
    file: StaticString = #filePath, line: UInt = #line
  ) throws {
    XCTAssertEqual(result.faceLandmarks.first?.count, 478, file: file, line: line)
    let landmark = result.faceLandmarks.first?.first
    XCTAssertEqual(
      landmark?.x ?? -1, firstLandmark.0, accuracy: Fixture.tolerance, file: file, line: line)
    XCTAssertEqual(
      landmark?.y ?? -1, firstLandmark.1, accuracy: Fixture.tolerance, file: file, line: line)
    let blendshape = result.faceBlendshapes.first?.categories.first
    XCTAssertEqual(blendshape?.categoryName, "_neutral", file: file, line: line)
    XCTAssertEqual(blendshape?.score ?? -1, 0, accuracy: 0.12, file: file, line: line)
    let matrix = try XCTUnwrap(result.facialTransformationMatrixes.first, file: file, line: line)
    XCTAssertEqual([matrix.rows, matrix.columns], [4, 4], file: file, line: line)
    for (row, expected) in matrixColumn0.enumerated() {
      let actual = matrix.value(atRow: row, column: 0)
      XCTAssertEqual(actual, expected, accuracy: Fixture.tolerance, file: file, line: line)
    }
  }

  func testOptionsMatrixAndConnectionsBridging() throws {
    let custom = FaceLandmarkerOptions(numFaces: 3, outputFaceBlendshapes: true)
    XCTAssertThrowsError(
      try FaceLandmarkerOptions(numFaces: 0).withCOptions(resultCallback: nil) { _ in })
    try custom.withCOptions(resultCallback: nil) { cOptions in
      XCTAssertEqual(cOptions.running_mode, VisionRunningMode.image.cRunningMode)
      XCTAssertEqual(cOptions.num_faces, 3)
      XCTAssertEqual(cOptions.min_tracking_confidence, 0.5)
      XCTAssertTrue(cOptions.output_face_blendshapes)
    }
    XCTAssertEqual(FaceLandmarker.rightIrisConnections, FaceLandmarksConnections.rightIris)
    XCTAssertEqual(FaceLandmarker.faceConnections, FaceLandmarksConnections.faceConnections)
    var values: [Float] = [1, 2, 3, 4, 5, 6]
    let matrix = values.withUnsafeMutableBufferPointer {
      TransformMatrix(cMatrix: MpMatrix(rows: 2, cols: 3, data: $0.baseAddress))
    }
    XCTAssertEqual(matrix, try TransformMatrix(data: [1, 2, 3, 4, 5, 6], rows: 2, columns: 3))
    XCTAssertEqual(matrix.value(atRow: 1, column: 0), 2)
    XCTAssertEqual(matrix.value(atRow: 0, column: 2), 5)

    XCTAssertThrowsError(try TransformMatrix(data: [1, 2, 3], rows: 2, columns: 3)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(try TransformMatrix(data: [], rows: -1, columns: 0)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(try TransformMatrix(data: [], rows: 0, columns: -1)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(try TransformMatrix(data: [], rows: .max, columns: 2)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
  }

  func testInitAndRunningModeValidation() throws {
    XCTAssertThrowsError(try FaceLandmarker(options: FaceLandmarkerOptions())) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
    }
    XCTAssertThrowsError(try FaceLandmarker(options: options(.liveStream)))
    let landmarker = try FaceLandmarker(options: options())
    XCTAssertThrowsError(
      try landmarker.detect(videoFrame: image("portrait"), timestampInMilliseconds: 0))
    XCTAssertThrowsError(
      try landmarker.detectAsync(image: image("portrait"), timestampInMilliseconds: 0))
  }

  func testDetectImageVideoAndRotatedImage() throws {
    let portrait = try image("portrait")
    let imageResult = try FaceLandmarker(options: options()).detect(image: portrait)
    try assertPortraitResult(
      imageResult, firstLandmark: (0.4977, 0.2485), matrixColumn0: [0.9991, 0.0166, -0.0374, 0])

    let rotated = try FaceLandmarker(options: options()).detect(
      image: image("portrait_rotated"),
      imageProcessingOptions: ImageProcessingOptions(rotationDegrees: -90))
    try assertPortraitResult(
      rotated, firstLandmark: (0.75075, 0.49812), matrixColumn0: [0.0212, -0.99878, -0.0374, 0])

    let videoLandmarker = try FaceLandmarker(options: options(.video))
    for timestamp in [0, 33] {
      let result = try videoLandmarker.detect(
        videoFrame: portrait, timestampInMilliseconds: timestamp)
      XCTAssertEqual(result.timestampInMilliseconds, timestamp)
    }
    XCTAssertThrowsError(
      try videoLandmarker.detect(videoFrame: portrait, timestampInMilliseconds: -1))
  }

  func testLiveStreamDeliversResultsToHandler() throws {
    let delivered = expectation(description: "live stream result")
    var liveOptions = try options(.liveStream)
    liveOptions.liveStreamResultHandler = { result, timestamp in
      XCTAssertEqual(timestamp, 42)
      if case .failure(let error) = result { XCTFail("Unexpected live stream error: \(error)") }
      XCTAssertEqual(try? result.get().faceLandmarks.first?.count, 478)
      delivered.fulfill()
    }
    let landmarker = try FaceLandmarker(options: liveOptions)
    try landmarker.detectAsync(image: image("portrait"), timestampInMilliseconds: 42)
    wait(for: [delivered], timeout: 10)
  }
}
