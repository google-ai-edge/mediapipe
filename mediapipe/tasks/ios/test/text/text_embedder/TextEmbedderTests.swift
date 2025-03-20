// Copyright 2023 The MediaPipe Authors.
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
import XCTest

@testable import MPPTextEmbedder

/// These tests are only for validating the Swift function signatures of the TextEmbedder.
/// Objective C tests of the TextEmbedder provide more coverage with unit tests for
/// different models and text embedder options. They can be found here:
/// /mediapipe/tasks/ios/test/text/text_embedder/MPPTextEmbedderTests.m

class TextEmbedderTests: XCTestCase {

  static let bundle = Bundle(for: TextEmbedderTests.self)

  static let bertModelPath = bundle.path(
    forResource: "mobilebert_embedding_with_metadata",
    ofType: "tflite")

  static let text1 = "it's a charming and often affecting journey"

  static let text2 = "what a great and fantastic trip"

  static let floatDiffTolerance: Float = 1e-4

  static let doubleDiffTolerance: Double = 1e-4

  func assertEqualErrorDescriptions(
    _ error: Error, expectedLocalizedDescription: String
  ) {
    XCTAssertEqual(
      error.localizedDescription,
      expectedLocalizedDescription)
  }

  func assertTextEmbedderResultHasOneEmbedding(
    _ textEmbedderResult: TextEmbedderResult
  ) {
    XCTAssertEqual(textEmbedderResult.embeddingResult.embeddings.count, 1)
  }

  func assertEmbeddingIsFloat(
    _ embedding: Embedding
  ) {
    XCTAssertNil(embedding.quantizedEmbedding)
    XCTAssertNotNil(embedding.floatEmbedding)
  }

  func assertEmbedding(
    _ floatEmbedding: [NSNumber],
    hasCount embeddingCount: Int,
    hasFirstValue firstValue: Float
  ) {
    XCTAssertEqual(floatEmbedding.count, embeddingCount)
    XCTAssertEqual(
      floatEmbedding[0].floatValue,
      firstValue,
      accuracy:
        TextEmbedderTests.floatDiffTolerance)
  }

  func assertFloatEmbeddingResultsForEmbed(
    text: String,
    using textEmbedder: TextEmbedder,
    hasCount embeddingCount: Int,
    hasFirstValue firstValue: Float
  ) throws -> Embedding {
    let textEmbedderResult =
      try XCTUnwrap(
        textEmbedder.embed(text: text))
    assertTextEmbedderResultHasOneEmbedding(textEmbedderResult)
    assertEmbeddingIsFloat(textEmbedderResult.embeddingResult.embeddings[0])
    assertEmbedding(
      textEmbedderResult.embeddingResult.embeddings[0].floatEmbedding!,
      hasCount: embeddingCount,
      hasFirstValue: firstValue)

    return textEmbedderResult.embeddingResult.embeddings[0]
  }

  func testEmbedWithBertSucceeds() throws {

    let modelPath = try XCTUnwrap(TextEmbedderTests.bertModelPath)
    let textEmbedder = try XCTUnwrap(TextEmbedder(modelPath: modelPath))

    let embedding1 = try assertFloatEmbeddingResultsForEmbed(
      text: TextEmbedderTests.text1,
      using: textEmbedder,
      hasCount: 512,
      hasFirstValue: 21.178507)

    let embedding2 = try assertFloatEmbeddingResultsForEmbed(
      text: TextEmbedderTests.text2,
      using: textEmbedder,
      hasCount: 512,
      hasFirstValue: 19.684338)

    let cosineSimilarity = try XCTUnwrap(
      TextEmbedder.cosineSimilarity(
        embedding1: embedding1,
        embedding2: embedding2))

    XCTAssertEqual(
      cosineSimilarity.doubleValue,
      0.96236,
      accuracy: TextEmbedderTests.doubleDiffTolerance)
  }
}
