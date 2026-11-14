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
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksUniversalEmbedder
#endif

/// Indexes text, images, audio, and composite `TaskPart` records in a `VectorStore` and retrieves
/// the highest-scoring matches for multimodal or precomputed vector queries.
///
/// - Note: Synchronize concurrent access externally (for example, inside a Swift `actor`) when
///   sharing an instance across tasks.
public final class SemanticRetriever {
  /// Default embedding vector dimensionality (`768`).
  public static let defaultEmbeddingDimension = SqliteVectorStore.defaultEmbeddingDimension

  private let components: SemanticRetrieverComponents

  /// Creates a retriever wired to custom storage, chunking, and embedding `components`.
  public init(components: SemanticRetrieverComponents) throws {
    self.components = components
  }

  /// Creates a retriever configured from `options` using `SqliteVectorStore`,
  /// `DefaultTextChunker`, and `options.embedder`.
  ///
  /// - Throws: `TasksError` if `options` are invalid or the SQLite database cannot be initialized.
  public convenience init(options: SemanticRetrieverOptions) throws {
    guard options.embeddingDimension > 0 else {
      throw TasksError.invalidArgument(
        "embeddingDimension must be positive, got \(options.embeddingDimension)."
      )
    }
    let chunker = try DefaultTextChunker(
      chunkSize: options.chunkSize,
      chunkOverlap: options.chunkOverlap,
      mode: options.chunkingMode
    )
    let store = SqliteVectorStore(
      databasePath: options.databasePath ?? ":memory:",
      embeddingDimension: options.embeddingDimension
    )
    try store.ensureInitialized()
    let components = try SemanticRetrieverComponents(
      vectorStore: store,
      chunker: chunker,
      providers: [options.embedder]
    )
    try self.init(components: components)
  }

  /// Chunks, embeds, and persists `text` under `id` with optional `metadata`.
  public func insertDocument(
    id: String,
    text: String,
    metadata: [String: String] = [:]
  ) throws {
    try insertContent(id: id, parts: [.text(text)], metadata: metadata)
  }

  /// Embeds and persists an image under `id`, encoding from `data` when provided and storing
  /// `filePath` as the persistent handle.
  public func insertImage(
    id: String,
    filePath: String,
    data: Data? = nil,
    metadata: [String: String] = [:]
  ) throws {
    guard !filePath.isEmpty else {
      throw TasksError.invalidArgument("filePath must not be empty.")
    }
    try insertContent(
      id: id,
      parts: [.image(ImagePart(filePath: filePath, data: data))],
      metadata: metadata
    )
  }

  /// Embeds and persists an audio clip under `id`, encoding from `samples` or `data` when provided
  /// and storing `filePath` as the persistent handle.
  public func insertAudio(
    id: String,
    filePath: String,
    data: Data? = nil,
    samples: [Float]? = nil,
    metadata: [String: String] = [:]
  ) throws {
    guard !filePath.isEmpty else {
      throw TasksError.invalidArgument("filePath must not be empty.")
    }
    try insertContent(
      id: id,
      parts: [.audio(AudioPart(filePath: filePath, data: data, samples: samples))],
      metadata: metadata
    )
  }

  /// Embeds and persists multimodal `parts` under `id`, replacing any prior record and chunks with
  /// the same identifier only after all embeddings succeed.
  public func insertContent(
    id: String,
    parts: [TaskPart],
    metadata: [String: String] = [:]
  ) throws {
    guard !id.isEmpty else {
      throw TasksError.invalidArgument("Record id must not be empty.")
    }
    guard !parts.isEmpty else {
      throw TasksError.invalidArgument("Content parts must not be empty.")
    }

    var recordsToUpsert: [RetrievalRecord] = []

    if parts.count == 1, case .text(let textPart) = parts[0] {
      let chunks = try components.chunker.chunk(text: textPart.text)
      if chunks.count > 1 {
        var childIds: [String] = []
        childIds.reserveCapacity(chunks.count)
        recordsToUpsert.reserveCapacity(chunks.count + 1)
        for (index, chunkText) in chunks.enumerated() {
          let childId = "\(id)_chunk_\(index)"
          childIds.append(childId)
          let chunkContent: [TaskPart] = [.text(chunkText)]
          let chunkEmbedding = try embedContent(chunkContent)
          recordsToUpsert.append(
            RetrievalRecord(
              id: childId,
              content: chunkContent,
              embeddings: chunkEmbedding,
              metadata: metadata,
              parentId: id
            )
          )
        }
        recordsToUpsert.append(
          RetrievalRecord(
            id: id,
            content: parts,
            embeddings: [],
            metadata: metadata,
            childIds: childIds
          )
        )
        try components.vectorStore.delete(ids: [id])
        try components.vectorStore.upsertRecords(recordsToUpsert)
        return
      }
    }

    let embedding = try embedContent(parts)
    recordsToUpsert.append(
      RetrievalRecord(
        id: id,
        content: parts,
        embeddings: embedding,
        metadata: metadata
      )
    )
    try components.vectorStore.delete(ids: [id])
    try components.vectorStore.upsertRecords(recordsToUpsert)
  }

  /// Returns up to `limit` top-level matches for `text` with cosine similarity at least
  /// `minSimilarity`, optionally filtered by `metadataFilter`.
  public func retrieve(
    text: String,
    limit: Int = 5,
    minSimilarity: Float = 0,
    metadataFilter: [String: String]? = nil
  ) throws -> [RetrievalResult] {
    try retrieve(
      parts: [.text(text)],
      limit: limit,
      minSimilarity: minSimilarity,
      metadataFilter: metadataFilter
    )
  }

  /// Returns up to `limit` top-level matches for multimodal `parts` with cosine similarity at
  /// least `minSimilarity`, optionally filtered by `metadataFilter`.
  public func retrieve(
    parts: [TaskPart],
    limit: Int = 5,
    minSimilarity: Float = 0,
    metadataFilter: [String: String]? = nil
  ) throws -> [RetrievalResult] {
    guard limit > 0 else {
      throw TasksError.invalidArgument("limit must be positive, got \(limit).")
    }
    let queryVector = try embedContent(parts)
    return try retrieve(
      embedding: queryVector,
      limit: limit,
      minSimilarity: minSimilarity,
      metadataFilter: metadataFilter
    )
  }

  /// Returns up to `limit` top-level matches for a precomputed query `embedding` with cosine
  /// similarity at least `minSimilarity`, optionally filtered by `metadataFilter`.
  public func retrieve(
    embedding: [Float],
    limit: Int = 5,
    minSimilarity: Float = 0,
    metadataFilter: [String: String]? = nil
  ) throws -> [RetrievalResult] {
    guard limit > 0 else {
      throw TasksError.invalidArgument("limit must be positive, got \(limit).")
    }
    guard !embedding.isEmpty else {
      throw TasksError.invalidArgument("Query embedding must not be empty.")
    }

    let searchLimit = limit.multipliedReportingOverflow(by: 3)
    let maxSearchLimit = max(limit, Int(Int32.max))
    let effectiveSearchLimit =
      searchLimit.overflow ? maxSearchLimit : min(searchLimit.partialValue, maxSearchLimit)

    let searchResults = try components.vectorStore.search(
      embedding: embedding,
      limit: effectiveSearchLimit,
      minSimilarity: minSimilarity,
      metadataFilter: metadataFilter
    )
    guard !searchResults.isEmpty else { return [] }

    let queryEmbedding = Embedding(floatEmbedding: embedding, headIndex: 0)
    var scoredCandidates: [(record: RetrievalRecord, score: Double)] = []
    scoredCandidates.reserveCapacity(searchResults.count)
    for candidate in searchResults {
      if candidate.similarityScore == nil && candidate.embeddings.isEmpty
        && !candidate.childIds.isEmpty
      {
        continue
      }
      let score = try Self.computeScore(for: candidate, queryEmbedding: queryEmbedding)
      if Float(score) >= minSimilarity {
        scoredCandidates.append((candidate, score))
      }
    }
    scoredCandidates.sort { $0.score > $1.score }

    var uniqueParentIds: [String] = []
    var seenParentLookup = Set<String>()
    for (record, _) in scoredCandidates {
      if let parentId = record.parentId, !parentId.isEmpty,
        seenParentLookup.insert(parentId).inserted
      {
        uniqueParentIds.append(parentId)
      }
    }

    var parentRecordsById: [String: RetrievalRecord] = [:]
    if !uniqueParentIds.isEmpty {
      for parent in try components.vectorStore.records(ids: uniqueParentIds) {
        parentRecordsById[parent.id] = parent
      }
    }

    var results: [RetrievalResult] = []
    var seenTopLevelIds = Set<String>()
    for (record, score) in scoredCandidates {
      if let parentId = record.parentId, !parentId.isEmpty {
        guard !seenTopLevelIds.contains(parentId),
          let parent = parentRecordsById[parentId]
        else {
          continue
        }
        seenTopLevelIds.insert(parentId)
        results.append(
          RetrievalResult(
            id: parent.id,
            content: parent.content,
            embeddings: record.embeddings,
            metadata: parent.metadata,
            score: score
          )
        )
      } else {
        guard seenTopLevelIds.insert(record.id).inserted else {
          continue
        }
        results.append(
          RetrievalResult(
            id: record.id,
            content: record.content,
            embeddings: record.embeddings,
            metadata: record.metadata,
            score: score
          )
        )
      }
      if results.count >= limit {
        break
      }
    }
    return results
  }

  /// Deletes the record stored under `id` along with any associated child chunks.
  public func delete(id: String) throws {
    try delete(ids: [id])
  }

  /// Deletes all records stored under `ids` along with any associated child chunks.
  public func delete(ids: [String]) throws {
    try components.vectorStore.delete(ids: ids)
  }

  /// Deletes all records matching every `metadataFilter` pair.
  ///
  /// - Throws: `TasksError` with `.invalidArgument` when `metadataFilter` is empty.
  public func delete(metadataFilter: [String: String]) throws {
    guard !metadataFilter.isEmpty else {
      throw TasksError.invalidArgument("metadataFilter cannot be empty.")
    }
    try components.vectorStore.delete(metadataFilter: metadataFilter)
  }

  /// Deletes every record in the store.
  public func deleteAllRecords() throws {
    try components.vectorStore.deleteAllRecords()
  }

  /// Returns all top-level record identifiers in the store.
  public func allRecordIds() throws -> [String] {
    try components.vectorStore.recordIdentifiers()
  }

  private func embedContent(_ content: [TaskPart]) throws -> [Float] {
    guard !content.isEmpty else {
      throw TasksError.invalidArgument("Content parts must not be empty.")
    }
    var lastError: (any Error)?
    for provider in components.providers {
      do {
        let result = try provider.embed(content: content)
        if let floats = result.embeddings.first?.floatEmbedding, !floats.isEmpty {
          return floats
        }
      } catch {
        lastError = error
      }
    }
    if let lastError {
      throw lastError
    }
    throw TasksError.failedPrecondition(
      "No embedding provider found that can embed this content."
    )
  }

  private static func computeScore(
    for record: RetrievalRecord,
    queryEmbedding: Embedding
  ) throws -> Double {
    if let similarityScore = record.similarityScore {
      return Double(similarityScore)
    }
    guard let queryFloats = queryEmbedding.floatEmbedding,
      record.embeddings.count == queryFloats.count
    else {
      throw TasksError.invalidArgument(
        "Stored record embedding dimension (\(record.embeddings.count)) does not match "
          + "query embedding dimension (\(queryEmbedding.floatEmbedding?.count ?? 0))."
      )
    }
    if record.embeddings.allSatisfy({ $0 == 0 }) || queryFloats.allSatisfy({ $0 == 0 }) {
      return 0
    }
    let stored = Embedding(floatEmbedding: record.embeddings, headIndex: 0)
    return try CosineSimilarity.compute(between: queryEmbedding, and: stored)
  }
}
