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
package import third_party_mediapipe_tasks_c_text_text_embedder_text_embedder

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
#endif

/// Output produced by a `TextEmbedder` inference call.
public struct TextEmbedderResult: Sendable, Equatable, Hashable {
  /// Embeddings produced across all output heads of the model.
  public let embeddingResult: EmbeddingResult

  /// Timestamp in milliseconds associated with this result.
  public let timestampInMilliseconds: Int

  /// Creates a result with the given embeddings and timestamp in milliseconds.
  public init(embeddingResult: EmbeddingResult, timestampInMilliseconds: Int = 0) {
    self.embeddingResult = embeddingResult
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  package init(cResult: MpTextEmbedderResult) {
    let embeddingResult = EmbeddingResult(cResult: cResult)
    self.init(
      embeddingResult: embeddingResult,
      timestampInMilliseconds: embeddingResult.timestampInMilliseconds ?? 0
    )
  }
}
