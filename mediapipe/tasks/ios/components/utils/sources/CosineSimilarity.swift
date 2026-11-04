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

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksCore
#endif

extension Embedding {
  /// Computes the cosine similarity between this embedding and `other` in `[-1.0, 1.0]`.
  ///
  /// - Parameter other: The embedding vector to compare against.
  /// - Throws: `TasksError` with `.invalidArgument` if the embeddings have mismatched types,
  ///   different vector lengths, are empty, or have a zero L2 norm.
  public func cosineSimilarity(with other: Embedding) throws -> Double {
    if let floats1 = floatEmbedding, let floats2 = other.floatEmbedding {
      return try Self.computeSimilarity(between: floats1, and: floats2) { Double($0) }
    }

    if let quantized1 = quantizedEmbedding, let quantized2 = other.quantizedEmbedding {
      return try Self.computeSimilarity(between: quantized1, and: quantized2) {
        Double(Int8(bitPattern: $0))
      }
    }

    throw TasksError.invalidArgument(
      "Cannot compute cosine similarity between quantized and float embeddings."
    )
  }

  private static func computeSimilarity<Element>(
    between vector1: [Element],
    and vector2: [Element],
    toDouble: (Element) -> Double
  ) throws -> Double {
    guard vector1.count == vector2.count else {
      throw TasksError.invalidArgument(
        "Cannot compute cosine similarity between embeddings of different sizes "
          + "(\(vector1.count) vs. \(vector2.count))."
      )
    }
    guard !vector1.isEmpty else {
      throw TasksError.invalidArgument("Cannot compute cosine similarity on empty embeddings.")
    }

    var dotProduct = 0.0
    var norm1 = 0.0
    var norm2 = 0.0

    for (rawU, rawV) in zip(vector1, vector2) {
      let u = toDouble(rawU)
      let v = toDouble(rawV)
      dotProduct += u * v
      norm1 += u * u
      norm2 += v * v
    }

    guard norm1 > 0.0, norm2 > 0.0 else {
      throw TasksError.invalidArgument(
        "Cannot compute cosine similarity on embedding with 0 norm."
      )
    }

    return dotProduct / (norm1.squareRoot() * norm2.squareRoot())
  }
}

/// Utility namespace for computing cosine similarity between `Embedding` values.
public enum CosineSimilarity {
  /// Computes the cosine similarity between `embedding1` and `embedding2` in `[-1.0, 1.0]`.
  public static func compute(
    between embedding1: Embedding,
    and embedding2: Embedding
  ) throws -> Double {
    try embedding1.cosineSimilarity(with: embedding2)
  }
}
