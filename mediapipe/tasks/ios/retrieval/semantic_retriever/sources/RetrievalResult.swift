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

/// Ranked match returned by a `SemanticRetriever` query.
public struct RetrievalResult: Sendable, Equatable, Hashable {
  /// Identifier of the matched top-level record.
  public let id: String

  /// Ordered multimodal segments of the matched top-level record.
  public let content: [TaskPart]

  /// Embedding vector of the highest-scoring matched record or chunk, if available.
  public let embeddings: [Float]?

  /// Key-value attributes attached to the matched top-level record.
  public let metadata: [String: String]

  /// Cosine similarity score between the query and the matched embedding.
  public let score: Double

  /// Creates a ranked retrieval match.
  public init(
    id: String,
    content: [TaskPart],
    embeddings: [Float]? = nil,
    metadata: [String: String] = [:],
    score: Double
  ) {
    self.id = id
    self.content = content
    self.embeddings = embeddings
    self.metadata = metadata
    self.score = score
  }
}
