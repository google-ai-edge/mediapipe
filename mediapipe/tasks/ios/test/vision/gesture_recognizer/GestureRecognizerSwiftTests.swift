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
import MediaPipeTasksGestureRecognizer
import Testing
import UIKit

private final class BundleToken {}

private final class LiveStreamCollector: @unchecked Sendable {
  private let lock = NSLock()
  private let sem = DispatchSemaphore(value: 0)
  private var stored: (Result<GestureRecognizerResult, any Error>, Int)?

  func record(_ outcome: Result<GestureRecognizerResult, any Error>, timestamp: Int) {
    lock.withLock { if stored == nil { stored = (outcome, timestamp) } }
    sem.signal()
  }

  func wait() -> Bool { sem.wait(timeout: .now() + 5.0) == .success }
  var firstEvent: (Result<GestureRecognizerResult, any Error>, Int)? { lock.withLock { stored } }
}

private enum Fixture {
  static let modelName = "gesture_recognizer"
  static let landmarkCount = 21
}

struct GestureRecognizerSwiftTests {
  private func resourcePath(
    _ name: String, ext: String, sourceLocation: SourceLocation = #_sourceLocation
  ) throws -> String {
    try #require(
      Bundle(for: BundleToken.self).path(forResource: name, ofType: ext),
      sourceLocation: sourceLocation)
  }

  private func loadImage(
    _ name: String, orientation: UIImage.Orientation = .up,
    sourceLocation: SourceLocation = #_sourceLocation
  ) throws -> MPImage {
    let path = try resourcePath(name, ext: "jpg", sourceLocation: sourceLocation)
    let uiImage = try #require(UIImage(contentsOfFile: path), sourceLocation: sourceLocation)
    return try MPImage(uiImage: uiImage, orientation: orientation)
  }

  private func makeRecognizer(
    mode: VisionRunningMode = .image, numHands: Int = 1,
    canned: ClassifierOptions = ClassifierOptions(),
    handler: GestureRecognizerOptions.LiveStreamResultHandler? = nil
  ) throws -> GestureRecognizer {
    let base = BaseOptions(modelAssetPath: try resourcePath(Fixture.modelName, ext: "task"))
    let opts = GestureRecognizerOptions(
      baseOptions: base, runningMode: mode, numHands: numHands,
      cannedGesturesClassifierOptions: canned, liveStreamResultHandler: handler)
    return try GestureRecognizer(options: opts)
  }

  private func expectInvalidArgument(
    sourceLocation: SourceLocation = #_sourceLocation,
    _ body: () throws -> some Any
  ) throws {
    let error = try #require(throws: TasksError.self, sourceLocation: sourceLocation) {
      _ = try body()
    }
    #expect(error.code == .invalidArgument, sourceLocation: sourceLocation)
  }

  private func expectSingleHand(
    _ result: GestureRecognizerResult, label: String,
    sourceLocation: SourceLocation = #_sourceLocation
  ) throws {
    let top = try #require(result.gestures.first?.first, sourceLocation: sourceLocation)
    #expect(result.gestures.count == 1, sourceLocation: sourceLocation)
    #expect(result.handedness.count == 1, sourceLocation: sourceLocation)
    #expect(top.index == -1, sourceLocation: sourceLocation)
    #expect(top.categoryName == label, sourceLocation: sourceLocation)
    #expect(top.score > 0.5, sourceLocation: sourceLocation)
    #expect(result.landmarks.first?.count == Fixture.landmarkCount, sourceLocation: sourceLocation)
    #expect(
      result.worldLandmarks.first?.count == Fixture.landmarkCount, sourceLocation: sourceLocation)
  }

  @Test func `options defaults and copy value semantics`() throws {
    let modelPath = try resourcePath(Fixture.modelName, ext: "task")
    let optsA = GestureRecognizerOptions(baseOptions: BaseOptions(modelAssetPath: modelPath))
    #expect(optsA.runningMode == .image)
    #expect(optsA.numHands == 1)
    #expect(optsA.minTrackingConfidence == 0.5)
    #expect(optsA.minHandDetectionConfidence == 0.5)
    #expect(optsA.minHandPresenceConfidence == 0.5)

    var optsB = optsA
    optsB.liveStreamResultHandler = { _, _ in }
    #expect(optsA.liveStreamResultHandler == nil)
    #expect(optsB.liveStreamResultHandler != nil)
  }

  @Test func `init rejects empty modelPath`() throws {
    try expectInvalidArgument { try GestureRecognizer(modelPath: "") }
  }

  @Test func `init rejects liveStream mode without result handler`() throws {
    try expectInvalidArgument { try makeRecognizer(mode: .liveStream) }
  }

  @Test func `init rejects non-liveStream mode with result handler`() throws {
    try expectInvalidArgument { try makeRecognizer(mode: .image) { _, _ in } }
  }

  @Test func `init rejects out-of-range numHands`() throws {
    #expect(throws: TasksError.self) { _ = try makeRecognizer(numHands: 0) }
    #expect(throws: TasksError.self) { _ = try makeRecognizer(numHands: .max) }
  }

  @Test func `recognize image identifies single-hand gestures and handedness`() throws {
    let modelPath = try resourcePath(Fixture.modelName, ext: "task")
    let recognizer = try GestureRecognizer(modelPath: modelPath)

    let thumbResult = try recognizer.recognize(image: try loadImage("thumb_up"))
    #expect(thumbResult.timestampInMilliseconds == 0)
    try expectSingleHand(thumbResult, label: "Thumb_Up")

    let fistResult = try recognizer.recognize(image: try loadImage("fist"))
    try expectSingleHand(fistResult, label: "Closed_Fist")
    #expect(fistResult.handedness.first?.first?.categoryName == "Right")
  }

  @Test func `recognize image returns empty result when no hands are present`() throws {
    let recognizer = try makeRecognizer()
    let empty = try recognizer.recognize(image: try loadImage("cats_and_dogs"))
    #expect(empty.gestures.isEmpty == true)
    #expect(empty.handedness.isEmpty == true)
    #expect(empty.landmarks.isEmpty == true)
  }

  @Test func `recognize image detects multiple hands when numHands is 2`() throws {
    let twoHands = try makeRecognizer(numHands: 2).recognize(image: try loadImage("right_hands"))
    #expect(twoHands.handedness.count == 2)
    #expect(twoHands.landmarks.count == 2)
  }

  @Test func `recognize image supports modelAssetBuffer and canned gesture filtering`() throws {
    let modelPath = try resourcePath(Fixture.modelName, ext: "task")
    let fist = try loadImage("fist")
    let modelData = try Data(contentsOf: URL(fileURLWithPath: modelPath))
    let allow = ClassifierOptions(scoreThreshold: 0.5, categoryAllowlist: ["Closed_Fist"])
    let bufOpts = GestureRecognizerOptions(
      baseOptions: BaseOptions(modelAssetBuffer: modelData), cannedGesturesClassifierOptions: allow)
    try expectSingleHand(
      try GestureRecognizer(options: bufOpts).recognize(image: fist), label: "Closed_Fist")

    let deny = ClassifierOptions(scoreThreshold: 0.5, categoryDenylist: ["Closed_Fist"])
    #expect(try makeRecognizer(canned: deny).recognize(image: fist).gestures.isEmpty == true)
  }

  @Test func `recognize image applies rotated UIImage orientation`() throws {
    let recognizer = try makeRecognizer()
    let rotated = try loadImage("pointing_up_rotated", orientation: .left)
    try expectSingleHand(try recognizer.recognize(image: rotated), label: "Pointing_Up")
  }

  @Test func `recognize rejects unsupported and out-of-bounds regionOfInterest`() throws {
    let recognizer = try makeRecognizer()
    let thumb = try loadImage("thumb_up")

    try expectInvalidArgument {
      try recognizer.recognize(
        image: thumb,
        imageProcessingOptions: ImageProcessingOptions(
          regionOfInterest: CGRect(x: 0.1, y: 0.1, width: 0.5, height: 0.5)))
    }
    try expectInvalidArgument {
      try recognizer.recognize(
        image: thumb,
        imageProcessingOptions: ImageProcessingOptions(
          regionOfInterest: CGRect(x: -0.1, y: 0.0, width: 0.5, height: 0.5)))
    }
  }

  @Test func `recognize rejects mismatched running modes`() throws {
    let thumb = try loadImage("thumb_up")

    let imageMode = try makeRecognizer(mode: .image)
    try expectInvalidArgument {
      try imageMode.recognize(videoFrame: thumb, timestampInMilliseconds: 0)
    }
    try expectInvalidArgument {
      try imageMode.recognizeAsync(image: thumb, timestampInMilliseconds: 0)
    }

    let videoMode = try makeRecognizer(mode: .video)
    try expectInvalidArgument { try videoMode.recognize(image: thumb) }
    try expectInvalidArgument {
      try videoMode.recognizeAsync(image: thumb, timestampInMilliseconds: 0)
    }

    let liveMode = try makeRecognizer(mode: .liveStream) { _, _ in }
    try expectInvalidArgument { try liveMode.recognize(image: thumb) }
    try expectInvalidArgument {
      try liveMode.recognize(videoFrame: thumb, timestampInMilliseconds: 0)
    }
  }

  @Test func `recognize video frames at monotonic timestamps`() throws {
    let thumb = try loadImage("thumb_up")
    let video = try makeRecognizer(mode: .video)

    let first = try video.recognize(videoFrame: thumb, timestampInMilliseconds: 0)
    #expect(first.timestampInMilliseconds == 0)
    try expectSingleHand(first, label: "Thumb_Up")

    let second = try video.recognize(videoFrame: thumb, timestampInMilliseconds: 33)
    #expect(second.timestampInMilliseconds == 33)
    try expectSingleHand(second, label: "Thumb_Up")
  }

  @Test func `recognize video rejects non-monotonic and out-of-range timestamps`() throws {
    let thumb = try loadImage("thumb_up")
    let video = try makeRecognizer(mode: .video)
    _ = try video.recognize(videoFrame: thumb, timestampInMilliseconds: 33)

    #expect(throws: TasksError.self) {
      _ = try video.recognize(videoFrame: thumb, timestampInMilliseconds: 10)
    }
    try expectInvalidArgument {
      try video.recognize(videoFrame: thumb, timestampInMilliseconds: -1)
    }
    try expectInvalidArgument {
      try video.recognize(videoFrame: thumb, timestampInMilliseconds: .max)
    }
  }

  @Test func `recognizeAsync in liveStream mode delivers result and rejects non-monotonic timestamps`()
    throws
  {
    let thumb = try loadImage("thumb_up")
    let collector = LiveStreamCollector()
    let live = try makeRecognizer(mode: .liveStream) { res, ts in
      collector.record(res, timestamp: ts)
    }
    try live.recognizeAsync(image: thumb, timestampInMilliseconds: 100)
    #expect(throws: TasksError.self) {
      try live.recognizeAsync(image: thumb, timestampInMilliseconds: 50)
    }
    #expect(collector.wait() == true)
    let event = try #require(collector.firstEvent)
    let liveResult = try event.0.get()
    #expect(event.1 == 100)
    #expect(liveResult.timestampInMilliseconds == 100)
    try expectSingleHand(liveResult, label: "Thumb_Up")
  }
}
