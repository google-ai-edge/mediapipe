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
import MediaPipeTasksHolisticLandmarker
import MediaPipeTasksVisionCore
import UIKit
import XCTest
import third_party_mediapipe_tasks_c_vision_core_image
import third_party_mediapipe_tasks_c_vision_holistic_landmarker_holistic_landmarker_lib

final class HolisticLandmarkerSwiftTests: XCTestCase {
  private static func resource(_ name: String, _ ext: String) throws -> String {
    let b = Bundle(for: HolisticLandmarkerSwiftTests.self)
    return try XCTUnwrap(b.path(forResource: name, ofType: ext))
  }

  private static func loadImage(_ name: String, orientation: UIImage.Orientation = .up) throws
    -> MPImage
  {
    let uiImage = try XCTUnwrap(UIImage(contentsOfFile: try resource(name, "jpg")))
    return try MPImage(uiImage: uiImage, orientation: orientation)
  }

  private func assertInvalidArg(
    file: StaticString = #filePath, line: UInt = #line, _ body: () throws -> some Any
  ) {
    XCTAssertThrowsError(try body(), file: file, line: line) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument, file: file, line: line)
    }
  }

  private static func assertHolisticLandmarks(
    _ r: HolisticLandmarkerResult, file: StaticString = #filePath, line: UInt = #line
  ) throws {
    let counts = [
      r.faceLandmarks.count, r.poseLandmarks.count, r.poseWorldLandmarks.count,
      r.leftHandLandmarks.count, r.leftHandWorldLandmarks.count,
      r.rightHandLandmarks.count, r.rightHandWorldLandmarks.count,
    ]
    XCTAssertEqual(counts, [478, 33, 33, 21, 21, 21, 21], file: file, line: line)
    let pose = try XCTUnwrap(r.poseLandmarks.first, file: file, line: line)
    let face = try XCTUnwrap(r.faceLandmarks.first, file: file, line: line)
    XCTAssertEqual(pose.x, 0.51091, accuracy: 0.03, file: file, line: line)
    XCTAssertEqual(pose.y, 0.15435, accuracy: 0.03, file: file, line: line)
    XCTAssertEqual(face.x, 0.51220, accuracy: 0.03, file: file, line: line)
    XCTAssertEqual(face.y, 0.18113, accuracy: 0.03, file: file, line: line)
  }

  func testOptionsValueSemanticsAndValidation() throws {
    let o = HolisticLandmarkerOptions()
    XCTAssertEqual(o.runningMode, .image)
    XCTAssertFalse(o.outputFaceBlendshapes || o.outputPoseSegmentationMasks)
    let fConf = [
      o.minFaceDetectionConfidence, o.minFaceSuppressionThreshold, o.minFacePresenceConfidence,
    ]
    let pConf = [
      o.minPoseDetectionConfidence, o.minPoseSuppressionThreshold, o.minPosePresenceConfidence,
    ]
    XCTAssertEqual(
      fConf + pConf + [o.minHandLandmarksConfidence], [0.5, 0.3, 0.5, 0.5, 0.3, 0.5, 0.5])
    XCTAssertNil(o.liveStreamResultHandler)

    var copy = o
    copy.runningMode = .video
    copy.minFaceSuppressionThreshold = 0.4
    copy.outputFaceBlendshapes = true
    copy.outputPoseSegmentationMasks = true
    XCTAssertEqual(o.runningMode, .image)
    try copy.withCOptions { cOpts in
      XCTAssertEqual(cOpts.running_mode, MP_RUNNING_MODE_VIDEO)
      XCTAssertTrue(cOpts.output_face_blendshapes && cOpts.output_pose_segmentation_masks)
      XCTAssertEqual(cOpts.min_face_suppression_threshold, 0.4, accuracy: 1e-6)
    }

    let manual = HolisticLandmarkerResult(
      faceLandmarks: [], poseLandmarks: [NormalizedLandmark(x: 0.2, y: 0.4, z: 0.1)],
      poseWorldLandmarks: [], leftHandLandmarks: [], leftHandWorldLandmarks: [],
      rightHandLandmarks: [], rightHandWorldLandmarks: [], timestampInMilliseconds: 99)
    XCTAssertEqual(manual, manual)

    XCTAssertThrowsError(try HolisticLandmarker(options: o)) {
      XCTAssertEqual(($0 as? TasksError)?.code, .invalidArgument)
      XCTAssertEqual(($0 as NSError).domain, TasksError.errorDomain)
    }
    let path = try Self.resource("holistic_landmarker", "task")
    let base = BaseOptions(modelAssetPath: path)
    let badLive = HolisticLandmarkerOptions(baseOptions: base, runningMode: .liveStream)
    let badImg = HolisticLandmarkerOptions(baseOptions: base, liveStreamResultHandler: { _, _ in })
    assertInvalidArg { try HolisticLandmarker(options: badLive) }
    assertInvalidArg { try HolisticLandmarker(options: badImg) }
  }

  func testImageVideoLiveStreamAndOrientationErrors() throws {
    let path = try Self.resource("holistic_landmarker", "task")
    let image = try Self.loadImage("male_full_height_hands")
    let landmarker = try HolisticLandmarker(modelPath: path)
    assertInvalidArg { try landmarker.detect(videoFrame: image, timestampInMilliseconds: 0) }
    assertInvalidArg { try landmarker.detectAsync(image: image, timestampInMilliseconds: 0) }
    let leftImg = try Self.loadImage("male_full_height_hands", orientation: .left)
    assertInvalidArg { try landmarker.detect(image: leftImg) }
    let rot90 = ImageProcessingOptions(rotationDegrees: 90)
    assertInvalidArg { try landmarker.detect(image: image, imageProcessingOptions: rot90) }
    let roi = ImageProcessingOptions(
      regionOfInterest: CGRect(x: 0.1, y: 0.1, width: 0.5, height: 0.5))
    assertInvalidArg { try landmarker.detect(image: image, imageProcessingOptions: roi) }

    let res0 = try landmarker.detect(image: image)
    try Self.assertHolisticLandmarks(res0)
    XCTAssertTrue(res0.faceBlendshapes == nil && res0.poseSegmentationMask == nil)

    let empty = try HolisticLandmarker(modelPath: path).detect(image: try Self.loadImage("cat"))
    XCTAssertTrue(
      empty.faceLandmarks.isEmpty && empty.faceBlendshapes == nil
        && empty.poseSegmentationMask == nil)

    let buf = try Data(contentsOf: URL(fileURLWithPath: path))
    let full = try HolisticLandmarker(
      options: .init(
        baseOptions: .init(modelAssetBuffer: buf), outputFaceBlendshapes: true,
        outputPoseSegmentationMasks: true))
    let fullRes = try full.detect(image: image)
    try Self.assertHolisticLandmarks(fullRes)
    XCTAssertEqual(fullRes.faceBlendshapes?.categories.count, 52)
    XCTAssertEqual(fullRes.faceBlendshapes?.categories.first?.categoryName, "_neutral")
    let mask = try XCTUnwrap(fullRes.poseSegmentationMask)
    XCTAssertTrue(
      mask.width == Int(image.width) && mask.height == Int(image.height)
        && mask.dataType == .float32)

    let video = try HolisticLandmarker(
      options: .init(baseOptions: .init(modelAssetPath: path), runningMode: .video))
    assertInvalidArg { try video.detect(videoFrame: image, timestampInMilliseconds: -1) }
    for ts in [0, 33] {
      let r = try video.detect(videoFrame: image, timestampInMilliseconds: ts)
      XCTAssertEqual(r.timestampInMilliseconds, ts)
      try Self.assertHolisticLandmarks(r)
    }
    assertInvalidArg { try video.detect(videoFrame: image, timestampInMilliseconds: 10) }

    let delivered = expectation(description: "live stream result")
    let live = try HolisticLandmarker(
      options: .init(
        baseOptions: .init(modelAssetPath: path), runningMode: .liveStream,
        liveStreamResultHandler: { outcome, ts in
          XCTAssertEqual(ts, 10)
          do {
            let liveRes = try outcome.get()
            XCTAssertEqual(liveRes.timestampInMilliseconds, 10)
            try Self.assertHolisticLandmarks(liveRes)
          } catch {
            XCTFail("Unexpected live stream error: \(error)")
          }
          delivered.fulfill()
        }))
    assertInvalidArg { try live.detectAsync(image: image, timestampInMilliseconds: -1) }
    try live.detectAsync(image: image, timestampInMilliseconds: 10)
    assertInvalidArg { try live.detectAsync(image: image, timestampInMilliseconds: 5) }
    wait(for: [delivered], timeout: 10.0)
  }
}
