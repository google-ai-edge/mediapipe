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

import CoreGraphics
import Foundation
import MediaPipeTasksComponents
import MediaPipeTasksCore
import MediaPipeTasksHandLandmarker
import MediaPipeTasksVisionCore
import UIKit
import XCTest

final class HandLandmarkerSwiftTests: XCTestCase {
  private enum Fixture {
    static let landmarkTolerance: Float = 0.03
    static let scoreTolerance: Float = 0.02
    static let landmarkCount = 21
  }

  private func resourcePath(_ name: String, _ ext: String = "jpg") throws -> String {
    try XCTUnwrap(Bundle(for: Self.self).path(forResource: name, ofType: ext))
  }

  private func loadImage(
    _ name: String = "thumb_up", orientation: UIImage.Orientation? = nil
  ) throws -> MPImage {
    let uiImage = try XCTUnwrap(UIImage(contentsOfFile: try resourcePath(name)))
    return try MPImage(uiImage: uiImage, orientation: orientation)
  }

  private func makeOptions(
    _ path: String, mode: VisionRunningMode = .image, numHands: Int = 1,
    handler: HandLandmarkerOptions.LiveStreamResultHandler? = nil
  ) -> HandLandmarkerOptions {
    HandLandmarkerOptions(
      baseOptions: BaseOptions(modelAssetPath: path), runningMode: mode, numHands: numHands,
      liveStreamResultHandler: handler)
  }

  private func assertThumbUp(
    _ r: HandLandmarkerResult, file: StaticString = #filePath, line: UInt = #line
  ) throws {
    let tol = Fixture.landmarkTolerance
    XCTAssertEqual([r.landmarks.count, r.worldLandmarks.count], [1, 1], file: file, line: line)
    XCTAssertEqual(r.landmarks[0].count, Fixture.landmarkCount, file: file, line: line)
    XCTAssertEqual(r.worldLandmarks[0].count, Fixture.landmarkCount, file: file, line: line)
    XCTAssertEqual(r.handedness.first?.first?.categoryName, "Right", file: file, line: line)
    XCTAssertEqual(
      try XCTUnwrap(r.handedness.first?.first?.score), 1.0, accuracy: Fixture.scoreTolerance,
      file: file, line: line)
    let wrist = r.landmarks[0][HandLandmark.wrist.rawValue]
    let tip = r.landmarks[0][HandLandmark.thumbTip.rawValue]
    let worldWrist = r.worldLandmarks[0][HandLandmark.wrist.rawValue]
    XCTAssertEqual(wrist.x, 0.6388, accuracy: tol, file: file, line: line)
    XCTAssertEqual(wrist.y, 0.6713, accuracy: tol, file: file, line: line)
    XCTAssertEqual(tip.x, 0.4736, accuracy: tol, file: file, line: line)
    XCTAssertEqual(tip.y, 0.2510, accuracy: tol, file: file, line: line)
    XCTAssertEqual(worldWrist.x, 0.0675, accuracy: tol, file: file, line: line)
    XCTAssertEqual(worldWrist.y, 0.0311, accuracy: tol, file: file, line: line)
  }

  func testOptionsValidationAndTopologyConnections() throws {
    let d = HandLandmarkerOptions()
    XCTAssertEqual(d.runningMode, .image)
    XCTAssertEqual(d.numHands, 1)
    XCTAssertEqual(
      [d.minHandDetectionConfidence, d.minHandPresenceConfidence, d.minTrackingConfidence],
      [0.5, 0.5, 0.5])
    XCTAssertEqual(HandLandmark.allCases.map(\.rawValue), Array(0..<Fixture.landmarkCount))
    let counts = [
      HandLandmarker.handPalmConnections, HandLandmarker.handThumbConnections,
      HandLandmarker.handIndexFingerConnections, HandLandmarker.handMiddleFingerConnections,
      HandLandmarker.handRingFingerConnections, HandLandmarker.handPinkyConnections,
      HandLandmarker.handConnections,
    ].map(\.count)
    XCTAssertEqual(counts, [6, 3, 3, 3, 3, 3, 21])

    XCTAssertThrowsError(try HandLandmarker(modelPath: ""))
    let path = try resourcePath("hand_landmarker", "task")
    XCTAssertThrowsError(try HandLandmarker(options: makeOptions(path, numHands: 0)))
    XCTAssertThrowsError(try HandLandmarker(options: makeOptions(path, mode: .liveStream)))
    XCTAssertThrowsError(try HandLandmarker(options: makeOptions(path, handler: { _, _ in })))
  }

  func testImageModeSingleMultiEmptyOrientationAndRoiRejection() throws {
    let path = try resourcePath("hand_landmarker", "task")
    let landmarker = try HandLandmarker(modelPath: path)
    let img = try loadImage()

    let single = try landmarker.detect(image: img)
    XCTAssertEqual(single.timestampInMilliseconds, 0)
    try assertThumbUp(single)
    XCTAssertEqual(single, Set([single]).first)

    let empty = try landmarker.detect(image: try loadImage("cats_and_dogs"))
    XCTAssertTrue(
      empty.landmarks.isEmpty && empty.worldLandmarks.isEmpty && empty.handedness.isEmpty)

    let rotated = try landmarker.detect(
      image: try loadImage("pointing_up_rotated", orientation: .right))
    XCTAssertEqual(rotated.landmarks.count, 1)
    XCTAssertEqual(rotated.landmarks[0][0].x, 0.2555, accuracy: Fixture.landmarkTolerance)
    XCTAssertEqual(rotated.landmarks[0][8].x, 0.7998, accuracy: Fixture.landmarkTolerance)

    let roi = ImageProcessingOptions(
      regionOfInterest: CGRect(x: 0.1, y: 0.1, width: 0.5, height: 0.5))
    XCTAssertThrowsError(try landmarker.detect(image: img, imageProcessingOptions: roi))
    XCTAssertThrowsError(try landmarker.detect(videoFrame: img, timestampInMilliseconds: 0))
    XCTAssertThrowsError(try landmarker.detectAsync(image: img, timestampInMilliseconds: 0))

    let multi = try HandLandmarker(options: makeOptions(path, numHands: 2))
    let multiRes = try multi.detect(image: try loadImage("right_hands"))
    XCTAssertEqual(
      [multiRes.handedness.count, multiRes.landmarks.count, multiRes.worldLandmarks.count],
      [2, 2, 2])
  }

  func testVideoAndLiveStreamModes() throws {
    let path = try resourcePath("hand_landmarker", "task")
    let img = try loadImage()

    let video = try HandLandmarker(options: makeOptions(path, mode: .video))
    XCTAssertThrowsError(try video.detect(image: img))
    XCTAssertThrowsError(try video.detect(videoFrame: img, timestampInMilliseconds: -1))
    for ts in [0, 300] {
      let result = try video.detect(videoFrame: img, timestampInMilliseconds: ts)
      XCTAssertEqual(result.timestampInMilliseconds, ts)
      try assertThumbUp(result)
    }

    let exp = expectation(description: "liveStream callback")
    final class CallbackState: @unchecked Sendable {
      let lock = NSLock()
      var received: (Result<HandLandmarkerResult, any Error>, Int)?
    }
    let state = CallbackState()
    let live = try HandLandmarker(
      options: makeOptions(path, mode: .liveStream) { outcome, ts in
        state.lock.withLock {
          if state.received == nil {
            state.received = (outcome, ts)
            exp.fulfill()
          }
        }
      })
    XCTAssertThrowsError(try live.detectAsync(image: img, timestampInMilliseconds: -1))
    try live.detectAsync(image: img, timestampInMilliseconds: 42)
    wait(for: [exp], timeout: 10.0)
    let pair = try XCTUnwrap(state.lock.withLock { state.received })
    let liveResult = try pair.0.get()
    XCTAssertEqual([pair.1, liveResult.timestampInMilliseconds], [42, 42])
    try assertThumbUp(liveResult)
  }
}
