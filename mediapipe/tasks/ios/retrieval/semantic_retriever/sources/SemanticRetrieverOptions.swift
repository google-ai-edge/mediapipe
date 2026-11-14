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
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksUniversalEmbedder
#endif

/// Convenience configuration for creating a `SemanticRetriever` backed by `SqliteVectorStore`,
/// `DefaultTextChunker`, and `UniversalEmbedder`.
public struct SemanticRetrieverOptions: Sendable {
  /// Multimodal embedder retained by the retriever to encode documents and queries.
  public var embedder: UniversalEmbedder

  /// Filesystem path to the SQLite database file, or `nil` for an in-memory database.
  public var databasePath: String?

  /// Dimensionality of the vectors produced by `embedder`.
  public var embeddingDimension: Int

  /// Size of each text chunk in characters or words depending on `chunkingMode`.
  public var chunkSize: Int

  /// Overlap between consecutive text chunks.
  public var chunkOverlap: Int

  /// Granularity used to partition inserted text documents.
  public var chunkingMode: ChunkingMode

  /// Creates options for a retriever backed by `embedder`.
  public init(
    embedder: UniversalEmbedder,
    databasePath: String? = nil,
    embeddingDimension: Int = SqliteVectorStore.defaultEmbeddingDimension,
    chunkSize: Int = 500,
    chunkOverlap: Int = 100,
    chunkingMode: ChunkingMode = .character
  ) {
    self.embedder = embedder
    self.databasePath = databasePath
    self.embeddingDimension = embeddingDimension
    self.chunkSize = chunkSize
    self.chunkOverlap = chunkOverlap
    self.chunkingMode = chunkingMode
  }
}
