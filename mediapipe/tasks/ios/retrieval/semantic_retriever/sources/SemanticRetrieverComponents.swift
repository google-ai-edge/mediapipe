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
  import MediaPipeTasksComponents
#endif

/// Pluggable storage, text-chunking, and embedding dependencies for `SemanticRetriever`.
public struct SemanticRetrieverComponents {
  /// Backend storing indexed records and performing vector similarity queries.
  public var vectorStore: any VectorStore

  /// Strategy used to partition single-part text documents prior to embedding.
  public var chunker: any TextChunker

  /// Ordered embedding providers queried to encode multimodal content.
  public var providers: [any EmbeddingProvider]

  /// Creates retriever dependencies, defaulting `chunker` to `DefaultTextChunker()` when `nil`.
  public init(
    vectorStore: any VectorStore,
    chunker: (any TextChunker)? = nil,
    providers: [any EmbeddingProvider]
  ) throws {
    self.vectorStore = vectorStore
    self.chunker = try chunker ?? DefaultTextChunker()
    self.providers = providers
  }
}
