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
import MediaPipeTasksImageClassifier
import Testing
import UIKit

private final class BundleToken {}

private final class LiveStreamCollector: @unchecked Sendable {
  private let lock = NSLock()
  private let semaphore = DispatchSemaphore(value: 0)
  private var stored: (Result<ImageClassifierResult, any Error>, Int)?

  func record(_ outcome: Result<ImageClassifierResult, any Error>, timestamp: Int) {
    lock.withLock { if stored == nil { stored = (outcome, timestamp) } }
    semaphore.signal()
  }

  func wait(timeout: TimeInterval = 5.0) -> Bool {
    semaphore.wait(timeout: .now() + timeout) == .success
  }

  var firstEvent: (Result<ImageClassifierResult, any Error>, Int)? { lock.withLock { stored } }
}

private enum Fixture {
  static let floatModel = "mobilenet_v2_1.0_224"
  static let quantizedModel = "mobilenet_v1_0.25_224_quant"
  static let categoryCount = 1001
  static let burgerTopThree: [(Int, Float, String)] = [
    (934, 0.786005, "cheeseburger"), (932, 0.023508, "bagel"), (925, 0.021172, "guacamole"),
  ]
}

struct ImageClassifierSwiftTests {
  private func resourcePath(_ name: String, ext: String) throws -> String {
    try #require(Bundle(for: BundleToken.self).path(forResource: name, ofType: ext))
  }

  private func loadImage(_ name: String, orientation: UIImage.Orientation = .up) throws -> MPImage {
    let uiImage = try #require(UIImage(contentsOfFile: try resourcePath(name, ext: "jpg")))
    return try MPImage(uiImage: uiImage, orientation: orientation)
  }

  private func makeClassifier(
    model: String = Fixture.floatModel, mode: VisionRunningMode = .image,
    maxResults: Int = -1,
    scoreThreshold: Float = 0.0,
    categoryAllowlist: [String] = [],
    categoryDenylist: [String] = [],
    handler: ImageClassifierOptions.LiveStreamResultHandler? = nil
  ) throws -> ImageClassifier {
    let base = BaseOptions(modelAssetPath: try resourcePath(model, ext: "tflite"))
    let opts = ImageClassifierOptions(
      baseOptions: base, runningMode: mode, maxResults: maxResults,
      scoreThreshold: scoreThreshold, categoryAllowlist: categoryAllowlist,
      categoryDenylist: categoryDenylist, liveStreamResultHandler: handler)
    return try ImageClassifier(options: opts)
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

  private func expectTop(
    _ result: ImageClassifierResult, count: Int, _ prefix: [(Int, Float, String)],
    sourceLocation: SourceLocation = #_sourceLocation
  ) throws {
    let classifications = result.classificationResult.classifications
    let head = try #require(classifications.first, sourceLocation: sourceLocation)
    #expect(classifications.count == 1, sourceLocation: sourceLocation)
    #expect(head.headIndex == 0, sourceLocation: sourceLocation)
    #expect(head.categories.count == count, sourceLocation: sourceLocation)
    for (actual, expected) in zip(head.categories, prefix) {
      #expect(actual.index == expected.0, sourceLocation: sourceLocation)
      #expect(actual.categoryName == expected.2, sourceLocation: sourceLocation)
      #expect(abs(actual.score - expected.1) < 1e-2, sourceLocation: sourceLocation)
    }
  }

  @Test func `options copy preserves value semantics`() throws {
    let modelPath = try resourcePath(Fixture.floatModel, ext: "tflite")
    let optsA = ImageClassifierOptions(baseOptions: BaseOptions(modelAssetPath: modelPath))
    var optsB = optsA
    optsB.liveStreamResultHandler = { _, _ in }
    #expect(optsA.liveStreamResultHandler == nil)
    #expect(optsB.liveStreamResultHandler != nil)
  }

  @Test func `init rejects empty modelPath`() throws {
    try expectInvalidArgument { try ImageClassifier(modelPath: "") }
  }

  @Test func `init rejects liveStream mode without result handler`() throws {
    try expectInvalidArgument { try makeClassifier(mode: .liveStream) }
  }

  @Test func `init rejects non-liveStream mode with result handler`() throws {
    try expectInvalidArgument { try makeClassifier(mode: .image) { _, _ in } }
  }

  @Test func `init rejects mutually exclusive allowlist and denylist`() throws {
    try expectInvalidArgument {
      try makeClassifier(categoryAllowlist: ["cheeseburger"], categoryDenylist: ["bagel"])
    }
  }

  @Test func `classify image with float model returns top categories`() throws {
    let floatPath = try resourcePath(Fixture.floatModel, ext: "tflite")
    let burger = try loadImage("burger")
    let fullResult = try ImageClassifier(modelPath: floatPath).classify(image: burger)
    #expect(fullResult.timestampInMilliseconds == 0)
    try expectTop(fullResult, count: Fixture.categoryCount, Fixture.burgerTopThree)
  }

  @Test func `classify image with modelAssetBuffer and allowlist filters categories`() throws {
    let floatPath = try resourcePath(Fixture.floatModel, ext: "tflite")
    let burger = try loadImage("burger")
    let modelData = try Data(contentsOf: URL(fileURLWithPath: floatPath))
    let bufferClassifier = try ImageClassifier(
      options: ImageClassifierOptions(
        baseOptions: BaseOptions(modelAssetBuffer: modelData), maxResults: 3,
        scoreThreshold: 0.25, categoryAllowlist: ["cheeseburger", "guacamole"]))
    let filtered = try bufferClassifier.classify(image: burger)
    try expectTop(filtered, count: 1, [(934, 0.786005, "cheeseburger")])
  }

  @Test func `classify image with categoryDenylist excludes denied category`() throws {
    let burger = try loadImage("burger")
    let denylist = try makeClassifier(maxResults: 2, categoryDenylist: ["bagel"])
    try expectTop(
      try denylist.classify(image: burger), count: 2,
      [(934, 0.786005, "cheeseburger"), (925, 0.021172, "guacamole")])
  }

  @Test func `classify image with quantized model returns top category`() throws {
    let burger = try loadImage("burger")
    let quant = try makeClassifier(model: Fixture.quantizedModel, maxResults: 1)
    try expectTop(try quant.classify(image: burger), count: 1, [(934, 0.96875, "cheeseburger")])
  }

  @Test func `classify image crops to regionOfInterest`() throws {
    let topOne = try makeClassifier(maxResults: 1)
    let soccerRoi = ImageProcessingOptions(
      regionOfInterest: CGRect(x: 0.450, y: 0.308, width: 0.164, height: 0.426))
    let soccer = try topOne.classify(
      image: try loadImage("multi_objects"), imageProcessingOptions: soccerRoi)
    try expectTop(soccer, count: 1, [(806, 0.997122, "soccer ball")])
  }

  @Test func `classify image applies rotated orientation and regionOfInterest`() throws {
    let topOne = try makeClassifier(maxResults: 1)
    let rotated = try topOne.classify(image: try loadImage("burger_rotated", orientation: .left))
    try expectTop(rotated, count: 1, [(934, 0.753852, "cheeseburger")])

    let chairRoi = ImageProcessingOptions(
      regionOfInterest: CGRect(x: 0.0, y: 0.1763, width: 0.5642, height: 0.1286))
    let chair = try topOne.classify(
      image: try loadImage("multi_objects_rotated", orientation: .left),
      imageProcessingOptions: chairRoi)
    try expectTop(chair, count: 1, [(560, 0.604605, "folding chair")])
  }

  @Test func `classify rejects out-of-bounds and empty regionOfInterest`() throws {
    let classifier = try makeClassifier()
    let burger = try loadImage("burger")
    try expectInvalidArgument {
      try classifier.classify(
        image: burger,
        imageProcessingOptions: ImageProcessingOptions(
          regionOfInterest: CGRect(x: -0.1, y: 0.0, width: 0.5, height: 0.5)))
    }
    try expectInvalidArgument {
      try classifier.classify(
        image: burger,
        imageProcessingOptions: ImageProcessingOptions(
          regionOfInterest: CGRect(x: 0.5, y: 0.5, width: 0.8, height: 0.8)))
    }
    try expectInvalidArgument {
      try classifier.classify(
        image: burger,
        imageProcessingOptions: ImageProcessingOptions(
          regionOfInterest: CGRect(x: 0.2, y: 0.2, width: 0.0, height: 0.4)))
    }
  }

  @Test func `classify rejects mismatched running modes`() throws {
    let burger = try loadImage("burger")

    let imageMode = try makeClassifier(mode: .image)
    try expectInvalidArgument {
      try imageMode.classify(videoFrame: burger, timestampInMilliseconds: 0)
    }
    try expectInvalidArgument {
      try imageMode.classifyAsync(image: burger, timestampInMilliseconds: 0)
    }

    let videoMode = try makeClassifier(mode: .video)
    try expectInvalidArgument { try videoMode.classify(image: burger) }
    try expectInvalidArgument {
      try videoMode.classifyAsync(image: burger, timestampInMilliseconds: 0)
    }

    let liveMode = try makeClassifier(mode: .liveStream) { _, _ in }
    try expectInvalidArgument { try liveMode.classify(image: burger) }
    try expectInvalidArgument {
      try liveMode.classify(videoFrame: burger, timestampInMilliseconds: 0)
    }
  }

  @Test func `classify video frames at monotonic timestamps`() throws {
    let burger = try loadImage("burger")
    let video = try makeClassifier(mode: .video, maxResults: 3)

    let first = try video.classify(videoFrame: burger, timestampInMilliseconds: 0)
    #expect(first.timestampInMilliseconds == 0)
    try expectTop(first, count: 3, Fixture.burgerTopThree)

    let second = try video.classify(videoFrame: burger, timestampInMilliseconds: 33)
    #expect(second.timestampInMilliseconds == 33)
    try expectTop(second, count: 3, Fixture.burgerTopThree)
  }

  @Test func `classify video rejects non-monotonic and out-of-range timestamps`() throws {
    let burger = try loadImage("burger")
    let video = try makeClassifier(mode: .video)
    _ = try video.classify(videoFrame: burger, timestampInMilliseconds: 66)

    #expect(throws: TasksError.self) {
      _ = try video.classify(videoFrame: burger, timestampInMilliseconds: 10)
    }
    try expectInvalidArgument {
      try video.classify(videoFrame: burger, timestampInMilliseconds: -1)
    }
    try expectInvalidArgument {
      try video.classify(videoFrame: burger, timestampInMilliseconds: .max)
    }
  }

  @Test func `classifyAsync in liveStream mode delivers result and rejects non-monotonic timestamps`()
    throws
  {
    let burger = try loadImage("burger")
    let collector = LiveStreamCollector()
    let live = try makeClassifier(mode: .liveStream, maxResults: 3) {
      outcome, ts in collector.record(outcome, timestamp: ts)
    }
    try live.classifyAsync(image: burger, timestampInMilliseconds: 100)
    #expect(throws: TasksError.self) {
      try live.classifyAsync(image: burger, timestampInMilliseconds: 50)
    }
    #expect(collector.wait() == true)
    let event = try #require(collector.firstEvent)
    let liveResult = try event.0.get()
    #expect(event.1 == 100)
    #expect(liveResult.timestampInMilliseconds == 100)
    try expectTop(liveResult, count: 3, Fixture.burgerTopThree)
  }
}
