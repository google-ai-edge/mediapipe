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
import MediaPipeTasksTextEmbedder
import XCTest

final class TextEmbedderSmokeTests: XCTestCase {

  func testOptionsCoverage() {
    var options = TextEmbedderOptions()
    options.l2Normalize = true
    options.quantize = false

    XCTAssertTrue(options.l2Normalize)
    XCTAssertFalse(options.quantize)

    var context = TextFormatContext()
    context.embeddingType = .retrievalQuery
    context.title = "Doc Title"
    context.textRole = .document
    XCTAssertEqual(context.embeddingType, .retrievalQuery)
    XCTAssertEqual(context.title, "Doc Title")
    XCTAssertEqual(context.textRole, .document)
  }

  func testRuntimeRegistrationAndInitialization() {
    var options = TextEmbedderOptions()
    options.baseOptions.modelAssetPath = "/nonexistent/text_embedder.task"

    XCTAssertThrowsError(try TextEmbedder(options: options)) { error in
      guard let tasksError = error as? TasksError else {
        XCTFail("Expected TasksError, got \(error)")
        return
      }
      XCTAssertTrue(tasksError.code == .invalidArgument || tasksError.code == .notFound)
      let nsError = error as NSError
      XCTAssertEqual(nsError.domain, TasksError.errorDomain)
      XCTAssertTrue(
        nsError.code == TasksErrorCode.invalidArgument.rawValue
          || nsError.code == TasksErrorCode.notFound.rawValue
      )
      XCTAssertTrue(
        error.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(error.localizedDescription)"
      )
    }

    XCTAssertThrowsError(
      try TextEmbedder(modelPath: "/nonexistent/text_embedder.task")
    ) { error in
      guard let tasksError = error as? TasksError else {
        XCTFail("Expected TasksError, got \(error)")
        return
      }
      XCTAssertTrue(tasksError.code == .invalidArgument || tasksError.code == .notFound)
      XCTAssertTrue(
        error.localizedDescription.contains("Unable to open file"),
        "Unexpected error: \(error.localizedDescription)"
      )
    }
  }

  func testExecutionMethodSignaturesCompile() {
    let _: (TextEmbedder, String) throws -> TextEmbedderResult = { embedder, text in
      try embedder.embed(text: text)
    }
    let _: (TextEmbedder) -> (String, TextFormatContext?) throws -> TextEmbedderResult =
      TextEmbedder.embed(text:textFormatContext:)
    let _: (Embedding, Embedding) throws -> Double =
      TextEmbedder.cosineSimilarity(embedding1:embedding2:)
  }

  func testResultContainerTypes() {
    let embedding = Embedding(
      floatEmbedding: [1.0, 0.0],
      quantizedEmbedding: nil,
      headIndex: 0,
      headName: "head"
    )
    XCTAssertEqual(embedding.floatEmbedding, [1.0, 0.0])
    XCTAssertNil(embedding.quantizedEmbedding)
    XCTAssertEqual(embedding.headIndex, 0)
    XCTAssertEqual(embedding.headName, "head")

    let embeddingResult = EmbeddingResult(
      embeddings: [embedding],
      timestampInMilliseconds: 1000
    )
    XCTAssertEqual(embeddingResult.embeddings.count, 1)
    XCTAssertEqual(embeddingResult.timestampInMilliseconds, 1000)

    let result = TextEmbedderResult(
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
      headIndex: 0,
      headName: "head"
    )
    let embedding2 = Embedding(
      floatEmbedding: [1.0, 0.0],
      quantizedEmbedding: nil,
      headIndex: 0,
      headName: "head"
    )

    let similarity = try TextEmbedder.cosineSimilarity(
      embedding1: embedding1,
      embedding2: embedding2
    )
    XCTAssertEqual(similarity, 1.0, accuracy: 1e-3)
  }
}
