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
package import third_party_mediapipe_tasks_c_components_containers_embedding_result

/// Feature vector produced by a single embedder output head.
public struct Embedding: Sendable, Equatable, Hashable {
  /// Floating-point embedding values, or `nil` when scalar quantization is enabled.
  public let floatEmbedding: [Float]?

  /// Scalar-quantized embedding bytes, or `nil` when scalar quantization is disabled.
  public let quantizedEmbedding: [UInt8]?

  /// The index of the embedder head that produced this embedding.
  public let headIndex: Int

  /// The optional tensor metadata name of the embedder head.
  public let headName: String?

  /// Creates an `Embedding` with the given floating-point or quantized values and head metadata.
  public init(
    floatEmbedding: [Float]? = nil,
    quantizedEmbedding: [UInt8]? = nil,
    headIndex: Int,
    headName: String? = nil
  ) {
    self.floatEmbedding = floatEmbedding
    self.quantizedEmbedding = quantizedEmbedding
    self.headIndex = headIndex
    self.headName = headName
  }

  /// Creates an `Embedding` copied from a C `MpEmbedding` struct.
  package init(cEmbedding: MpEmbedding) {
    let count = max(0, Int(cEmbedding.values_count))
    if let floatPtr = cEmbedding.float_embedding {
      self.floatEmbedding = Array(UnsafeBufferPointer(start: floatPtr, count: count))
      self.quantizedEmbedding = nil
    } else if let quantizedPtr = cEmbedding.quantized_embedding {
      self.floatEmbedding = nil
      self.quantizedEmbedding = UnsafeBufferPointer(start: quantizedPtr, count: count).map {
        UInt8(bitPattern: $0)
      }
    } else {
      self.floatEmbedding = nil
      self.quantizedEmbedding = nil
    }

    self.headIndex = Int(cEmbedding.head_index)
    self.headName = cEmbedding.head_name.map { String(cString: $0) }
  }
}

/// Embedding results produced across all output heads of an embedding model.
public struct EmbeddingResult: Sendable, Equatable, Hashable {
  /// Embeddings produced for each head of the model.
  public let embeddings: [Embedding]

  /// Timestamp in milliseconds of the start of the data chunk corresponding to these results, or
  /// `nil` when no timestamp is associated with the result.
  public let timestampInMilliseconds: Int?

  /// Creates an `EmbeddingResult` with the given head embeddings and optional timestamp.
  public init(embeddings: [Embedding], timestampInMilliseconds: Int? = nil) {
    self.embeddings = embeddings
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  /// Creates an `EmbeddingResult` copied from a C `MpEmbeddingResult` struct.
  package init(cResult: MpEmbeddingResult) {
    self.embeddings = Array(
      cArray: cResult.embeddings,
      count: cResult.embeddings_count,
      transform: Embedding.init
    )
    self.timestampInMilliseconds = cResult.has_timestamp_ms ? Int(cResult.timestamp_ms) : nil
  }
}
