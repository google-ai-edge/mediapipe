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

import MPPCommon
import MPPImageEmbedder
import XCTest

extension TasksErrorCode {
  fileprivate static let errorDomain = "com.google.mediapipe.tasks"
}

final class ImageEmbedderSmokeTests: XCTestCase {

  private final class MockLiveStreamDelegate: NSObject,
    ImageEmbedderLiveStreamDelegate
  {
    func imageEmbedder(
      _ imageEmbedder: ImageEmbedder,
      didFinishEmbedding result: ImageEmbedderResult?,
      timestampInMilliseconds: Int,
      error: (any Error)?
    ) {}
  }

  func testOptionsAndDelegateCoverage() {
    let options = ImageEmbedderOptions()
    options.runningMode = .liveStream
    options.l2Normalize = true
    options.quantize = false

    let delegate = MockLiveStreamDelegate()
    options.imageEmbedderLiveStreamDelegate = delegate
    XCTAssertNotNil(options.imageEmbedderLiveStreamDelegate)
    XCTAssertEqual(options.runningMode, .liveStream)
    XCTAssertTrue(options.l2Normalize)
    XCTAssertFalse(options.quantize)
  }

  func testRuntimeRegistrationAndInitialization() {
    let options = ImageEmbedderOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/embedder.task"

    XCTAssertThrowsError(try ImageEmbedder(options: options)) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
      XCTAssertFalse(
        nsError.localizedDescription.contains("No registered object with name"),
        "ImageEmbedder subgraph not linked"
      )
      XCTAssertTrue(
        nsError.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(nsError.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try ImageEmbedder(modelPath: "/nonexistent/embedder.task")
    ) { error in
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksErrorCode.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgumentError.rawValue
          || nsError.code == TasksErrorCode.notFoundError.rawValue
      )
    }
  }

  func testExecutionMethodSignaturesCompile() {
    let _: (ImageEmbedder) -> (MPImage) throws -> ImageEmbedderResult =
      ImageEmbedder.embed(image:)
    let _: (ImageEmbedder) -> (MPImage, CGRect) throws -> ImageEmbedderResult =
      ImageEmbedder.embed(image:regionOfInterest:)
    let _: (ImageEmbedder) -> (MPImage, Int) throws -> ImageEmbedderResult =
      ImageEmbedder.embed(videoFrame:timestampInMilliseconds:)
    let _: (ImageEmbedder) -> (MPImage, Int, CGRect) throws -> ImageEmbedderResult =
      ImageEmbedder.embed(videoFrame:timestampInMilliseconds:regionOfInterest:)
    let _: (ImageEmbedder) -> (MPImage, Int) throws -> Void =
      ImageEmbedder.embedAsync(image:timestampInMilliseconds:)
    let _: (ImageEmbedder) -> (MPImage, Int, CGRect) throws -> Void =
      ImageEmbedder.embedAsync(image:timestampInMilliseconds:regionOfInterest:)
    let _: (Embedding, Embedding) throws -> NSNumber =
      ImageEmbedder.cosineSimilarity(embedding1:embedding2:)
  }

  func testResultContainerTypes() {
    let embedding = Embedding(
      floatEmbedding: [0.1, 0.2, 0.3],
      quantizedEmbedding: nil,
      head: 0,
      headName: "head"
    )
    XCTAssertEqual(embedding.floatEmbedding, [0.1, 0.2, 0.3])
    XCTAssertNil(embedding.quantizedEmbedding)
    XCTAssertEqual(embedding.headIndex, 0)
    XCTAssertEqual(embedding.headName, "head")

    let embeddingResult = EmbeddingResult(
      embeddings: [embedding],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(embeddingResult.embeddings.count, 1)
    XCTAssertEqual(embeddingResult.timestampInMilliseconds, 1000)

    let result = ImageEmbedderResult(
      embeddingResult: embeddingResult,
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(result.embeddingResult.embeddings.count, 1)
    XCTAssertEqual(result.timestampInMilliseconds, 1000)
  }

  func testCosineSimilarity() throws {
    let embedding1 = Embedding(
      floatEmbedding: [1.0, 0.0],
      quantizedEmbedding: nil,
      head: 0,
      headName: "head"
    )
    let embedding2 = Embedding(
      floatEmbedding: [1.0, 0.0],
      quantizedEmbedding: nil,
      head: 0,
      headName: "head"
    )

    let similarity = try ImageEmbedder.cosineSimilarity(
      embedding1: embedding1,
      embedding2: embedding2
    )
    XCTAssertEqual(similarity.doubleValue, 1.0, accuracy: 1e-3)
  }
}
