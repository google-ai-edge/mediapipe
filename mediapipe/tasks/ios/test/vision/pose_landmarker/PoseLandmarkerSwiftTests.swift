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
import MediaPipeTasksPoseLandmarker
import Testing
import UIKit

private final class BundleToken {}

struct PoseLandmarkerSwiftTests {
  private static func resourcePath(_ fileName: String) throws -> String {
    let nsName = fileName as NSString
    return try #require(
      Bundle(for: BundleToken.self).path(
        forResource: nsName.deletingPathExtension, ofType: nsName.pathExtension))
  }

  private static func loadMPImage(_ fileName: String) throws -> MPImage {
    let uiImage = try #require(UIImage(contentsOfFile: try resourcePath(fileName)))
    return try MPImage(uiImage: uiImage)
  }

  private static func assertPoseMatchesExpected(
    _ result: PoseLandmarkerResult, sourceLocation: SourceLocation = #_sourceLocation
  ) throws {
    let pose = try #require(result.landmarks.first, sourceLocation: sourceLocation)
    let worldPose = try #require(result.worldLandmarks.first, sourceLocation: sourceLocation)
    let nose = try #require(pose.first, sourceLocation: sourceLocation)
    let worldNose = try #require(worldPose.first, sourceLocation: sourceLocation)
    #expect(result.landmarks.count == 1, sourceLocation: sourceLocation)
    #expect(result.worldLandmarks.count == 1, sourceLocation: sourceLocation)
    #expect(pose.count == 33, sourceLocation: sourceLocation)
    #expect(worldPose.count == 33, sourceLocation: sourceLocation)
    #expect(abs(nose.x - 0.4649) < 0.1, sourceLocation: sourceLocation)
    #expect(abs(nose.y - 0.4228) < 0.1, sourceLocation: sourceLocation)
    #expect(abs(nose.z - (-0.1500)) < 0.1, sourceLocation: sourceLocation)
    #expect(nose.visibility != nil, sourceLocation: sourceLocation)
    #expect(nose.presence != nil, sourceLocation: sourceLocation)
    #expect(abs(worldNose.x - (-0.0852)) < 0.1, sourceLocation: sourceLocation)
    #expect(abs(worldNose.y - (-0.6153)) < 0.1, sourceLocation: sourceLocation)
  }

  @Test
  func `options validation and connections`() throws {
    let defaults = PoseLandmarkerOptions()
    #expect(defaults.runningMode == .image)
    #expect(defaults.numPoses == 1)
    #expect(defaults.minPoseDetectionConfidence == 0.5)
    #expect(defaults.minPosePresenceConfidence == 0.5)
    #expect(defaults.minTrackingConfidence == 0.5)
    #expect(defaults.outputSegmentationMasks == false)
    #expect(defaults.liveStreamResultHandler == nil)
    #expect(PoseLandmarker.poseLandmarks.count == 35)
    #expect(PoseLandmarker.poseLandmarks.first == Connection(start: 0, end: 1))
    #expect(PoseLandmarker.poseLandmarks.last == Connection(start: 28, end: 32))

    let base = BaseOptions(modelAssetPath: try Self.resourcePath("pose_landmarker.task"))
    #expect(throws: TasksError.self) { _ = try PoseLandmarker(options: defaults) }
    #expect(throws: TasksError.self) {
      _ = try PoseLandmarker(options: PoseLandmarkerOptions(baseOptions: base, numPoses: 0))
    }
    #expect(throws: TasksError.self) {
      _ = try PoseLandmarker(
        options: PoseLandmarkerOptions(baseOptions: base, runningMode: .liveStream))
    }
    #expect(throws: TasksError.self) {
      _ = try PoseLandmarker(
        options: PoseLandmarkerOptions(baseOptions: base, liveStreamResultHandler: { _, _ in }))
    }
  }

  @Test
  func `detect image with and without segmentation masks`() throws {
    let modelPath = try Self.resourcePath("pose_landmarker.task")
    let poseImage = try Self.loadMPImage("pose.jpg")
    let landmarker = try PoseLandmarker(modelPath: modelPath)
    let result = try landmarker.detect(image: poseImage)
    try Self.assertPoseMatchesExpected(result)
    #expect(result.segmentationMasks == nil)
    #expect(result.timestampInMilliseconds == 0)
    _ = try landmarker.detect(
      image: poseImage, imageProcessingOptions: ImageProcessingOptions(rotationDegrees: 90))
    #expect(throws: TasksError.self) {
      _ = try landmarker.detect(
        image: poseImage,
        imageProcessingOptions: ImageProcessingOptions(
          regionOfInterest: CGRect(x: 0.1, y: 0.1, width: 0.5, height: 0.5)))
    }
    #expect(throws: TasksError.self) {
      _ = try landmarker.detect(videoFrame: poseImage, timestampInMilliseconds: 0)
    }

    let modelData = try Data(contentsOf: URL(fileURLWithPath: modelPath))
    let maskLandmarker = try PoseLandmarker(
      options: PoseLandmarkerOptions(
        baseOptions: BaseOptions(modelAssetBuffer: modelData), outputSegmentationMasks: true))
    let maskResult = try maskLandmarker.detect(image: poseImage)
    try Self.assertPoseMatchesExpected(maskResult)
    let masks = try #require(maskResult.segmentationMasks)
    let mask = try #require(masks.first)
    #expect(masks.count == 1)
    #expect(mask.width == 1000 && mask.height == 667)
    #expect(mask.dataType == .float32)

    let empty = try maskLandmarker.detect(image: try Self.loadMPImage("burger.jpg"))
    #expect(empty.landmarks.isEmpty)
    #expect(empty.worldLandmarks.isEmpty)
    #expect(empty.segmentationMasks == [])
  }

  @Test
  func `detect video and live stream modes`() throws {
    let base = BaseOptions(modelAssetPath: try Self.resourcePath("pose_landmarker.task"))
    let poseImage = try Self.loadMPImage("pose.jpg")
    let videoLandmarker = try PoseLandmarker(
      options: PoseLandmarkerOptions(baseOptions: base, runningMode: .video))
    for ts in [0, 33, 66] {
      let res = try videoLandmarker.detect(videoFrame: poseImage, timestampInMilliseconds: ts)
      try Self.assertPoseMatchesExpected(res)
      #expect(res.timestampInMilliseconds == ts)
    }
    #expect(throws: TasksError.self) {
      _ = try videoLandmarker.detect(videoFrame: poseImage, timestampInMilliseconds: -1)
    }
    #expect(throws: TasksError.self) {
      _ = try videoLandmarker.detect(videoFrame: poseImage, timestampInMilliseconds: .max)
    }
    #expect(throws: TasksError.self) {
      _ = try videoLandmarker.detect(videoFrame: poseImage, timestampInMilliseconds: 33)
    }

    let semaphore = DispatchSemaphore(value: 0)
    let lock = NSLock()
    nonisolated(unsafe) var delivered: (Result<PoseLandmarkerResult, any Error>, Int)?
    let streamLandmarker = try PoseLandmarker(
      options: PoseLandmarkerOptions(
        baseOptions: base, runningMode: .liveStream, outputSegmentationMasks: true,
        liveStreamResultHandler: { outcome, ts in
          lock.withLock {
            if delivered == nil {
              delivered = (outcome, ts)
              semaphore.signal()
            }
          }
        }))
    #expect(throws: TasksError.self) { _ = try streamLandmarker.detect(image: poseImage) }
    #expect(throws: TasksError.self) {
      try streamLandmarker.detectAsync(image: poseImage, timestampInMilliseconds: -1)
    }
    try streamLandmarker.detectAsync(image: poseImage, timestampInMilliseconds: 42)
    #expect(semaphore.wait(timeout: .now() + .seconds(10)) == .success)
    let (outcome, callbackTs) = try #require(lock.withLock { delivered })
    let streamResult = try outcome.get()
    try Self.assertPoseMatchesExpected(streamResult)
    #expect(streamResult.segmentationMasks?.count == 1)
    #expect(callbackTs == 42)
  }
}
