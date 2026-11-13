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

import Foundation
import MediaPipeTasksTextEmbedder
import XCTest

/// Unit tests for `TextEmbedder` across MobileBERT, Gecko, and EmbeddingGemma models.
final class TextEmbedderTests: XCTestCase {

  static let bundle = Bundle(for: TextEmbedderTests.self)

  static let bertModelPath = bundle.path(
    forResource: "mobilebert_embedding_with_metadata",
    ofType: "tflite")

  static let text1 = "it's a charming and often affecting journey"

  static let text2 = "what a great and fantastic trip"

  static let floatDiffTolerance: Float = 1e-4

  static let doubleDiffTolerance: Double = 1e-4

  static let cosineSimilarityThreshold: Double = 0.95

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
    _ floatEmbedding: [Float],
    hasCount embeddingCount: Int,
    hasFirstValue firstValue: Float
  ) {
    XCTAssertEqual(floatEmbedding.count, embeddingCount)
    XCTAssertEqual(
      floatEmbedding[0],
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
    let textEmbedderResult = try textEmbedder.embed(text: text)
    assertTextEmbedderResultHasOneEmbedding(textEmbedderResult)
    assertEmbeddingIsFloat(textEmbedderResult.embeddingResult.embeddings[0])
    let floatEmbedding = try XCTUnwrap(
      textEmbedderResult.embeddingResult.embeddings[0].floatEmbedding)
    assertEmbedding(
      floatEmbedding,
      hasCount: embeddingCount,
      hasFirstValue: firstValue)

    return textEmbedderResult.embeddingResult.embeddings[0]
  }

  func testEmbedWithBertSucceeds() throws {

    let modelPath = try XCTUnwrap(TextEmbedderTests.bertModelPath)
    let textEmbedder = try TextEmbedder(modelPath: modelPath)

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

    let cosineSimilarity = try TextEmbedder.cosineSimilarity(
      embedding1: embedding1,
      embedding2: embedding2)

    XCTAssertGreaterThanOrEqual(
      cosineSimilarity, TextEmbedderTests.cosineSimilarityThreshold)
  }

  func testEmbedWithGeckoSucceeds() throws {
    let modelPath = try XCTUnwrap(
      TextEmbedderTests.bundle.path(forResource: "gecko", ofType: "task"))
    let textEmbedder = try TextEmbedder(modelPath: modelPath)

    var textFormatContext = TextFormatContext()
    textFormatContext.embeddingType = .retrievalQuery
    textFormatContext.textRole = .query

    let result = try textEmbedder.embed(
      text: TextEmbedderTests.text1,
      textFormatContext: textFormatContext)

    assertTextEmbedderResultHasOneEmbedding(result)
    XCTAssertEqual(result.embeddingResult.embeddings[0].floatEmbedding?.count, 768)
  }

  func testEmbedWithEmbeddingGemmaSucceeds() throws {
    let modelPath = try XCTUnwrap(
      TextEmbedderTests.bundle.path(forResource: "embedding_gemma", ofType: "task"))
    let textEmbedder = try TextEmbedder(modelPath: modelPath)

    var textFormatContext = TextFormatContext()
    textFormatContext.embeddingType = .retrievalQuery
    textFormatContext.textRole = .query

    let result1 = try textEmbedder.embed(
      text: TextEmbedderTests.text1,
      textFormatContext: textFormatContext)

    assertTextEmbedderResultHasOneEmbedding(result1)
    XCTAssertEqual(result1.embeddingResult.embeddings[0].floatEmbedding?.count, 768)

    let result2 = try textEmbedder.embed(
      text: TextEmbedderTests.text2,
      textFormatContext: textFormatContext)

    assertTextEmbedderResultHasOneEmbedding(result2)
    XCTAssertEqual(result2.embeddingResult.embeddings[0].floatEmbedding?.count, 768)

    let cosineSimilarity = try TextEmbedder.cosineSimilarity(
      embedding1: result1.embeddingResult.embeddings[0],
      embedding2: result2.embeddingResult.embeddings[0])

    XCTAssertEqual(cosineSimilarity, 0.52, accuracy: 0.05)
  }

  func testEmbedWithL2NormalizeAndQuantizeOptionsSucceeds() throws {
    let modelPath = try XCTUnwrap(TextEmbedderTests.bertModelPath)

    var normalizedOptions = TextEmbedderOptions()
    normalizedOptions.baseOptions.modelAssetPath = modelPath
    normalizedOptions.l2Normalize = true
    let normalizedEmbedder = try TextEmbedder(options: normalizedOptions)
    let normalizedResult = try normalizedEmbedder.embed(text: TextEmbedderTests.text1)
    let normalizedValues = try XCTUnwrap(
      normalizedResult.embeddingResult.embeddings.first?.floatEmbedding)
    let normSquared = normalizedValues.reduce(Float.zero) { $0 + $1 * $1 }
    XCTAssertEqual(normSquared, 1.0, accuracy: TextEmbedderTests.floatDiffTolerance)

    var quantizedOptions = normalizedOptions
    quantizedOptions.quantize = true
    let quantizedEmbedder = try TextEmbedder(options: quantizedOptions)
    let quantizedResult = try quantizedEmbedder.embed(text: TextEmbedderTests.text1)
    let quantizedEmbedding = try XCTUnwrap(quantizedResult.embeddingResult.embeddings.first)
    XCTAssertNil(quantizedEmbedding.floatEmbedding)
    XCTAssertEqual(quantizedEmbedding.quantizedEmbedding?.count, 512)
  }
}
