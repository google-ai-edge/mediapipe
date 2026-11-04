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
import MediaPipeTasksComponents
import MediaPipeTasksCore
import Testing
import third_party_mediapipe_tasks_c_components_containers_embedding_result
import third_party_mediapipe_tasks_c_components_processors_classifier_options
import third_party_mediapipe_tasks_c_components_processors_embedder_options

struct ProcessorsAndEmbeddingsTests {
  @Test
  func `ClassifierOptions bridges string arrays and thresholds to MpClassifierOptions`() throws {
    let options = ClassifierOptions(
      displayNamesLocale: "en-US",
      maxResults: 5,
      scoreThreshold: 0.75,
      categoryAllowlist: ["cat", "dog", "bird"],
      categoryDenylist: ["unknown"]
    )

    try options.withCOptions { cOptions in
      let locale = cOptions.display_names_locale.map { String(cString: $0) }
      #expect(locale == "en-US")
      #expect(cOptions.max_results == 5)
      #expect(cOptions.score_threshold == 0.75)
      #expect(cOptions.category_allowlist_count == 3)
      #expect(cOptions.category_denylist_count == 1)

      let allowlistPtr = try #require(cOptions.category_allowlist)
      let allowlist = (0..<Int(cOptions.category_allowlist_count)).compactMap { index in
        allowlistPtr[index].map { String(cString: $0) }
      }
      #expect(allowlist == ["cat", "dog", "bird"])

      let denylistPtr = try #require(cOptions.category_denylist)
      let denylist = (0..<Int(cOptions.category_denylist_count)).compactMap { index in
        denylistPtr[index].map { String(cString: $0) }
      }
      #expect(denylist == ["unknown"])
    }

    let defaults = ClassifierOptions()
    defaults.withCOptions { cOptions in
      #expect(cOptions.display_names_locale == nil)
      #expect(cOptions.max_results == -1)
      #expect(cOptions.score_threshold == 0.0)
      #expect(cOptions.category_allowlist == nil)
      #expect(cOptions.category_allowlist_count == 0)
      #expect(cOptions.category_denylist == nil)
      #expect(cOptions.category_denylist_count == 0)
    }
  }

  @Test
  func `EmbedderOptions bridges boolean flags to MpEmbedderOptions`() {
    EmbedderOptions().withCOptions { cOptions in
      #expect(cOptions.l2_normalize == false)
      #expect(cOptions.quantize == false)
    }

    EmbedderOptions(l2Normalize: true, quantize: true).withCOptions { cOptions in
      #expect(cOptions.l2_normalize == true)
      #expect(cOptions.quantize == true)
    }
  }

  @Test
  func `EmbeddingResult bridges float, quantized, and empty C embeddings`() throws {
    var floats: [Float] = [0.6, 0.8]
    var quantized: [CChar] = [CChar(bitPattern: 0x7F), CChar(bitPattern: 0x80)]
    var headName = Array("embedder".utf8CString)

    try floats.withUnsafeMutableBufferPointer { floatBuf in
      try quantized.withUnsafeMutableBufferPointer { quantizedBuf in
        try headName.withUnsafeMutableBufferPointer { headNameBuf in
          var cEmbeddings = [
            MpEmbedding(
              float_embedding: floatBuf.baseAddress, quantized_embedding: nil,
              values_count: UInt32(floatBuf.count), head_index: 0,
              head_name: headNameBuf.baseAddress
            ),
            MpEmbedding(
              float_embedding: nil, quantized_embedding: quantizedBuf.baseAddress,
              values_count: UInt32(quantizedBuf.count), head_index: 1, head_name: nil
            ),
            MpEmbedding(
              float_embedding: floatBuf.baseAddress, quantized_embedding: nil,
              values_count: 0, head_index: 2, head_name: nil
            ),
          ]

          try cEmbeddings.withUnsafeMutableBufferPointer { embeddingsBuf in
            let cResult = MpEmbeddingResult(
              embeddings: embeddingsBuf.baseAddress,
              embeddings_count: UInt32(embeddingsBuf.count),
              timestamp_ms: 0,
              has_timestamp_ms: false
            )

            let result = EmbeddingResult(cResult: cResult)
            #expect(result.timestampInMilliseconds == nil)
            #expect(result.embeddings.count == 3)
            let first = try #require(result.embeddings.first)
            #expect(first.floatEmbedding == [0.6, 0.8])
            #expect(first.quantizedEmbedding == nil)
            #expect(first.headName == "embedder")
            #expect(result.embeddings[1].quantizedEmbedding == [0x7F, 0x80])
            #expect(result.embeddings[1].floatEmbedding == nil)
            #expect(result.embeddings[2].floatEmbedding == [])
            #expect(result.embeddings[2].quantizedEmbedding == nil)
          }
        }
      }
    }
  }

  @Test
  func `CosineSimilarity computes float and quantized similarity and validates inputs`() throws {
    let floatA = Embedding(floatEmbedding: [1.0, 0.0], headIndex: 0)
    let floatB = Embedding(floatEmbedding: [1.0, 0.0], headIndex: 0)
    #expect(abs(try floatA.cosineSimilarity(with: floatB) - 1.0) < 1e-6)
    #expect(abs(try CosineSimilarity.compute(between: floatA, and: floatB) - 1.0) < 1e-6)

    let quantA = Embedding(quantizedEmbedding: [127, 0], headIndex: 0)
    let quantB = Embedding(quantizedEmbedding: [UInt8(bitPattern: -127), 0], headIndex: 0)
    #expect(abs(try quantA.cosineSimilarity(with: quantB) - (-1.0)) < 1e-6)

    let mismatchedSize = Embedding(floatEmbedding: [1.0, 2.0, 3.0], headIndex: 0)
    #expect(throws: TasksError.self) {
      _ = try floatA.cosineSimilarity(with: mismatchedSize)
    }

    #expect(throws: TasksError.self) {
      _ = try floatA.cosineSimilarity(with: quantA)
    }

    let emptyFloat = Embedding(floatEmbedding: [], headIndex: 0)
    #expect(throws: TasksError.self) {
      _ = try emptyFloat.cosineSimilarity(with: emptyFloat)
    }

    let zeroNorm = Embedding(floatEmbedding: [0.0, 0.0], headIndex: 0)
    #expect(throws: TasksError.self) {
      _ = try floatA.cosineSimilarity(with: zeroNorm)
    }
  }
}
