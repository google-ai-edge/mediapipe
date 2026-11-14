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
import MediaPipeTasksImageEmbedder
import Testing
import UIKit

private final class BundleToken {}

private final class LiveStreamCollector: @unchecked Sendable {
  private let lock = NSLock()
  private let semaphore = DispatchSemaphore(value: 0)
  private var items: [(Result<ImageEmbedderResult, any Error>, Int)] = []

  func record(_ result: Result<ImageEmbedderResult, any Error>, timestamp: Int) {
    lock.withLock { items.append((result, timestamp)) }
    semaphore.signal()
  }

  func waitFor(count: Int) -> Bool {
    let deadline = DispatchTime.now() + 5.0
    for _ in 0..<count where semaphore.wait(timeout: deadline) == .timedOut { return false }
    return true
  }

  var snapshot: [(Result<ImageEmbedderResult, any Error>, Int)] { lock.withLock { items } }
}

private enum TestFixture {
  static let model = "mobilenet_v3_small_100_224_embedder"
  static let burger = "burger"
  static let burgerCrop = "burger_crop"
  static let burgerRotated = "burger_rotated"
  static let dimension = 1024
  static let tolerance = 1e-4
}

struct ImageEmbedderSwiftTests {
  private func path(_ name: String, _ ext: String) throws -> String {
    try #require(Bundle(for: BundleToken.self).path(forResource: name, ofType: ext))
  }

  private func loadImage(
    _ name: String, orientation: UIImage.Orientation? = nil
  ) throws -> MPImage {
    let uiImage = try #require(UIImage(contentsOfFile: try path(name, "jpg")))
    return try MPImage(uiImage: uiImage, orientation: orientation)
  }

  private func expectError(
    sourceLocation: SourceLocation = #_sourceLocation, _ body: () throws -> some Any
  ) {
    #expect(throws: TasksError.self, sourceLocation: sourceLocation) { _ = try body() }
  }

  private func opts(
    _ model: String, mode: VisionRunningMode = .image,
    l2Normalize: Bool = false,
    quantize: Bool = false,
    handler: ImageEmbedderOptions.LiveStreamResultHandler? = nil
  ) -> ImageEmbedderOptions {
    ImageEmbedderOptions(
      baseOptions: BaseOptions(modelAssetPath: model), runningMode: mode,
      l2Normalize: l2Normalize, quantize: quantize, liveStreamResultHandler: handler)
  }

  private func checkHead(
    _ res: ImageEmbedderResult, quantized: Bool = false,
    sourceLocation: SourceLocation = #_sourceLocation
  ) throws -> Embedding {
    #expect(res.embeddingResult.embeddings.count == 1, sourceLocation: sourceLocation)
    let head = try #require(res.embeddingResult.embeddings.first, sourceLocation: sourceLocation)
    let (present, absent) =
      quantized
      ? (head.quantizedEmbedding?.count, head.floatEmbedding?.count)
      : (head.floatEmbedding?.count, head.quantizedEmbedding?.count)
    #expect(present == TestFixture.dimension && absent == nil, sourceLocation: sourceLocation)
    return head
  }

  @Test func `Options validation and cosine similarity`() throws {
    let model = try path(TestFixture.model, "tflite")
    expectError { try ImageEmbedder(modelPath: "") }
    expectError { try ImageEmbedder(options: opts(model, mode: .liveStream)) }
    let withHandler = opts(model) { _, _ in }
    expectError { try ImageEmbedder(options: withHandler) }
    expectError { try ImageEmbedder(options: opts(model, mode: .video) { _, _ in }) }
    #expect(withHandler.liveStreamResultHandler != nil)

    let f1 = Embedding(floatEmbedding: [1, 0], headIndex: 0)
    let f2 = Embedding(floatEmbedding: [0, 1], headIndex: 0)
    let q1 = Embedding(quantizedEmbedding: [127, 0], headIndex: 0)
    #expect(abs(try ImageEmbedder.cosineSimilarity(embedding1: f1, embedding2: f1) - 1.0) < 1e-5)
    #expect(abs(try ImageEmbedder.cosineSimilarity(embedding1: f1, embedding2: f2)) < 1e-5)
    expectError { try ImageEmbedder.cosineSimilarity(embedding1: f1, embedding2: q1) }
    expectError {
      try ImageEmbedder.cosineSimilarity(
        embedding1: f1, embedding2: Embedding(floatEmbedding: [1], headIndex: 0))
    }
    expectError {
      try ImageEmbedder.cosineSimilarity(
        embedding1: Embedding(floatEmbedding: [0, 0], headIndex: 0), embedding2: f1)
    }
  }

  @Test func `Image embedding supports quantize, normalize, ROI, and orientation`() throws {
    let model = try path(TestFixture.model, "tflite")
    let burger = try loadImage(TestFixture.burger)
    let crop = try loadImage(TestFixture.burgerCrop)
    let rotated = try loadImage(TestFixture.burgerRotated, orientation: .left)

    let embedder = try ImageEmbedder(modelPath: model)
    let r1 = try embedder.embed(image: burger)
    let r2 = try embedder.embed(image: crop)
    #expect(r1.timestampInMilliseconds == 0)
    let (e1, e2) = (try checkHead(r1), try checkHead(r2))
    #expect(
      abs(try ImageEmbedder.cosineSimilarity(embedding1: e1, embedding2: e2) - 0.928711)
        < TestFixture.tolerance)

    let roiOpts = ImageProcessingOptions(
      regionOfInterest: CGRect(x: 0, y: 0, width: 0.833333, height: 1))
    let eRoi = try checkHead(try embedder.embed(image: burger, imageProcessingOptions: roiOpts))
    #expect(
      abs(try ImageEmbedder.cosineSimilarity(embedding1: eRoi, embedding2: e2) - 0.99992)
        < TestFixture.tolerance)

    let eRot = try checkHead(try embedder.embed(image: rotated))
    #expect(
      abs(try ImageEmbedder.cosineSimilarity(embedding1: eRot, embedding2: e1) - 0.98086)
        < TestFixture.tolerance)

    let rotRoi = ImageProcessingOptions(
      regionOfInterest: CGRect(x: 0, y: 0, width: 1, height: 0.833333))
    let eRotRoi = try checkHead(try embedder.embed(image: rotated, imageProcessingOptions: rotRoi))
    #expect(
      abs(try ImageEmbedder.cosineSimilarity(embedding1: eRotRoi, embedding2: e2) - 0.97749)
        < TestFixture.tolerance)

    expectError {
      try embedder.embed(image: burger, imageProcessingOptions: .init(rotationDegrees: 45))
    }
    expectError { try embedder.embed(videoFrame: burger, timestampInMilliseconds: 0) }
    expectError { try embedder.embedAsync(image: burger, timestampInMilliseconds: 0) }
    try embedder.close()
    expectError { try embedder.embed(image: burger) }

    let qEmb = try ImageEmbedder(options: opts(model, quantize: true))
    let (q1, q2) = (
      try checkHead(try qEmb.embed(image: burger), quantized: true),
      try checkHead(try qEmb.embed(image: crop), quantized: true)
    )
    #expect(
      abs(try ImageEmbedder.cosineSimilarity(embedding1: q1, embedding2: q2) - 0.92883)
        < TestFixture.tolerance)

    let data = try Data(contentsOf: URL(fileURLWithPath: model))
    let l2Opts = ImageEmbedderOptions(
      baseOptions: BaseOptions(modelAssetBuffer: data), l2Normalize: true)
    let l2Head = try checkHead(try ImageEmbedder(options: l2Opts).embed(image: burger))
    let normSq = try #require(l2Head.floatEmbedding).reduce(Float.zero) { $0 + $1 * $1 }
    #expect(abs(normSq - 1.0) < 1e-4)
  }

  @Test func `Video and liveStream modes enforce timestamps and async delivery`() throws {
    let model = try path(TestFixture.model, "tflite")
    let burger = try loadImage(TestFixture.burger)

    let video = try ImageEmbedder(options: opts(model, mode: .video))
    expectError { try video.embed(image: burger) }
    expectError { try video.embedAsync(image: burger, timestampInMilliseconds: 0) }
    for ts in [0, 33, 66] {
      let res = try video.embed(videoFrame: burger, timestampInMilliseconds: ts)
      #expect(res.timestampInMilliseconds == ts)
      _ = try checkHead(res)
    }
    expectError { try video.embed(videoFrame: burger, timestampInMilliseconds: 33) }
    expectError { try video.embed(videoFrame: burger, timestampInMilliseconds: -1) }

    let box = LiveStreamCollector()
    let live = try ImageEmbedder(
      options: opts(model, mode: .liveStream) { box.record($0, timestamp: $1) })
    expectError { try live.embed(image: burger) }
    expectError { try live.embed(videoFrame: burger, timestampInMilliseconds: 0) }
    try live.embedAsync(image: burger, timestampInMilliseconds: 10)
    expectError { try live.embedAsync(image: burger, timestampInMilliseconds: 5) }
    try live.close()
    try #require(box.waitFor(count: 1) == true)
    let (liveOutcome, cbTs) = try #require(box.snapshot.first)
    let liveRes = try liveOutcome.get()
    #expect(cbTs == 10 && liveRes.timestampInMilliseconds == 10)
    _ = try checkHead(liveRes)
  }
}
