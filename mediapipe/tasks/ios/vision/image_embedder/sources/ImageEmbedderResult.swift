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
package import third_party_mediapipe_tasks_c_vision_image_embedder_image_embedder

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksComponents
#endif

/// Feature vectors produced across all output heads of an image embedding model.
public struct ImageEmbedderResult: Sendable, Equatable, Hashable {
  /// Per-head feature vectors extracted from the input image or frame.
  public let embeddingResult: EmbeddingResult

  /// Frame timestamp in milliseconds (`0` in `.image` mode).
  public let timestampInMilliseconds: Int

  /// Creates a result from head embeddings and frame start time in milliseconds.
  public init(embeddingResult: EmbeddingResult, timestampInMilliseconds: Int = 0) {
    self.embeddingResult = embeddingResult
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  package init(cResult: MpImageEmbedderResult, timestampInMilliseconds: Int = 0) {
    let parsed = EmbeddingResult(cResult: cResult)
    self.embeddingResult = parsed
    self.timestampInMilliseconds = parsed.timestampInMilliseconds ?? timestampInMilliseconds
  }
}
