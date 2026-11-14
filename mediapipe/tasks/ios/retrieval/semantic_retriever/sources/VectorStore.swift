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
internal import third_party_mediapipe_tasks_c_core_mp_status
internal import third_party_mediapipe_tasks_c_retrieval_semantic_retriever_semantic_retriever

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksComponents
  import MediaPipeTasksCore
#endif

/// Storage backend for vector embeddings, multimodal part handles, and record metadata.
public protocol VectorStore {
  /// Inserts or replaces `records` in the store.
  func upsertRecords(_ records: [RetrievalRecord]) throws

  /// Searches for up to `limit` records whose similarity to `embedding` is at least
  /// `minSimilarity`, optionally restricted to records matching every `metadataFilter` pair.
  func search(
    embedding: [Float],
    limit: Int,
    minSimilarity: Float,
    metadataFilter: [String: String]?
  ) throws -> [RetrievalRecord]

  /// Returns all stored records matching `ids`.
  func records(ids: [String]) throws -> [RetrievalRecord]

  /// Returns the stored record matching `id`, or `nil` if not found.
  func record(id: String) throws -> RetrievalRecord?

  /// Returns all top-level record identifiers in the store (excluding child chunk or part rows).
  func recordIdentifiers() throws -> [String]

  /// Deletes all records matching `ids`, along with any associated child rows.
  func delete(ids: [String]) throws

  /// Deletes all records matching every `metadataFilter` pair, along with associated child rows.
  func delete(metadataFilter: [String: String]) throws

  /// Deletes every record in the store.
  func deleteAllRecords() throws
}

extension VectorStore {
  /// Searches for up to `limit` nearest records with a minimum similarity threshold of `0`.
  public func search(
    embedding: [Float],
    limit: Int = 5,
    metadataFilter: [String: String]? = nil
  ) throws -> [RetrievalRecord] {
    try search(
      embedding: embedding,
      limit: limit,
      minSimilarity: 0,
      metadataFilter: metadataFilter
    )
  }

  /// Searches for up to `limit` nearest records whose similarity is at least `minSimilarity`.
  public func search(
    embedding: [Float],
    limit: Int = 5,
    minSimilarity: Float
  ) throws -> [RetrievalRecord] {
    try search(
      embedding: embedding,
      limit: limit,
      minSimilarity: minSimilarity,
      metadataFilter: nil
    )
  }

  /// Searches for up to 5 nearest records matching `minSimilarity` and `metadataFilter`.
  public func search(
    embedding: [Float],
    minSimilarity: Float,
    metadataFilter: [String: String]?
  ) throws -> [RetrievalRecord] {
    try search(
      embedding: embedding,
      limit: 5,
      minSimilarity: minSimilarity,
      metadataFilter: metadataFilter
    )
  }

  /// Returns the stored record matching `id`, or `nil` if not found.
  public func record(id: String) throws -> RetrievalRecord? {
    try records(ids: [id]).first
  }
}

/// SQLite-backed `VectorStore` using native SQLite cosine-similarity indexing.
///
/// `@unchecked Sendable` is safe because the lazy SQLite handle `cStore` and all native database
/// operations are serialized through `lock`.
///
/// - Note: `upsertRecords(_:)` validates all input records upfront, but deleting superseded record
///   IDs and pruning stale child chunks before batch insertion is not atomic across a subsequent
///   SQLite I/O failure.
public final class SqliteVectorStore: @unchecked Sendable {
  /// Default embedding vector dimensionality (`768`).
  public static let defaultEmbeddingDimension = 768

  /// Filesystem path to the SQLite database file, or `":memory:"` / `""` for an in-memory store.
  public let databasePath: String

  /// Expected vector dimensionality for stored and queried embeddings.
  public let embeddingDimension: Int

  private let lock = NSLock()
  private var cStore: MpSqliteVectorStorePtr?

  /// Creates a SQLite vector store whose underlying database connection opens lazily on first use.
  public init(
    databasePath: String = ":memory:",
    embeddingDimension: Int = SqliteVectorStore.defaultEmbeddingDimension
  ) {
    self.databasePath = databasePath
    self.embeddingDimension = embeddingDimension
  }

  deinit {
    if let cStore {
      _ = MpSqliteVectorStoreClose(cStore, nil)
    }
  }

  func ensureInitialized() throws {
    try lock.withLock {
      _ = try ensureStoreUnlocked()
    }
  }

  private func ensureStoreUnlocked() throws -> MpSqliteVectorStorePtr {
    if let cStore {
      return cStore
    }
    guard embeddingDimension > 0, let cDim = Int32(exactly: embeddingDimension) else {
      throw TasksError.invalidArgument("Embedding dimension must be greater than 0.")
    }
    var created: MpSqliteVectorStorePtr?
    try databasePath.withCString { pathPtr in
      try MpStatus.check(defaultMessage: "Failed to initialize SqliteVectorStore.") { errorMsg in
        MpSqliteVectorStoreCreate(pathPtr, cDim, &created, &errorMsg)
      }
    }
    guard let created else {
      throw TasksError.internalError("MpSqliteVectorStoreCreate returned a null handle.")
    }
    self.cStore = created
    return created
  }

  private static func cCount(_ count: Int) throws -> Int32 {
    guard let cCount = Int32(exactly: count) else {
      throw TasksError.invalidArgument("Element count \(count) exceeds Int32.max.")
    }
    return cCount
  }
}

extension SqliteVectorStore: VectorStore {
  /// Inserts or replaces `records` after validating all inputs upfront.
  ///
  /// - Note: Deleting superseded record IDs and pruning stale child chunks precedes the batch
  ///   insert transaction and is not atomic across a subsequent SQLite I/O failure.
  public func upsertRecords(_ records: [RetrievalRecord]) throws {
    guard !records.isEmpty else { return }
    let recordCount = try Self.cCount(records.count)
    let arena = CRecordArena()
    defer { arena.deallocateAll() }

    var cRecords: [MpVectorStoreRecord] = []
    cRecords.reserveCapacity(records.count)

    for record in records {
      let partCount = try Self.cCount(record.content.count)
      let embCount = try Self.cCount(record.embeddings.count)
      let metaCount = try Self.cCount(record.metadata.count)

      let cParts: [MpTaskPart] = record.content.map { part in
        var cPart = MpTaskPart()
        switch part {
        case .text(let textPart):
          cPart.kind = kMpTaskPartText
          cPart.text_part.text = arena.cString(textPart.text)
        case .image(let imagePart):
          cPart.kind = kMpTaskPartImage
          cPart.image_part.file_path = imagePart.filePath.map { arena.cString($0) }
        case .audio(let audioPart):
          cPart.kind = kMpTaskPartAudio
          cPart.audio_part.audio_path = audioPart.filePath.map { arena.cString($0) }
        }
        return cPart
      }

      let cMeta: [MpKeyValuePair] = record.metadata.map { key, value in
        MpKeyValuePair(key: arena.cString(key), value: arena.cString(value))
      }

      let cChildIdPtrs: UnsafePointer<UnsafePointer<CChar>?>?
      let cChildIdsCount: Int32
      if !record.childIds.isEmpty {
        cChildIdsCount = try Self.cCount(record.childIds.count)
        let ptrArray: [UnsafePointer<CChar>?] = record.childIds.map { Optional(arena.cString($0)) }
        cChildIdPtrs = arena.copyBuffer(ptrArray)
      } else {
        cChildIdPtrs = nil
        cChildIdsCount = 0
      }

      var cRecord = MpVectorStoreRecord()
      cRecord.id = arena.cString(record.id)
      cRecord.parts = arena.copyBuffer(cParts)
      cRecord.parts_count = partCount
      cRecord.embeddings = arena.copyBuffer(record.embeddings)
      cRecord.embeddings_count = embCount
      if let score = record.similarityScore {
        cRecord.has_similarity_score = true
        cRecord.similarity_score = score
      }
      cRecord.metadata = arena.copyBuffer(cMeta)
      cRecord.metadata_count = metaCount
      cRecord.parent_id = record.parentId.map { arena.cString($0) }
      cRecord.child_ids = cChildIdPtrs
      cRecord.child_ids_count = cChildIdsCount
      cRecords.append(cRecord)
    }

    try cRecords.withUnsafeBufferPointer { recordsBuf in
      try lock.withLock {
        let store = try ensureStoreUnlocked()
        try MpStatus.check(defaultMessage: "Failed to upsert records.") { errorMsg in
          MpSqliteVectorStoreUpsert(store, recordsBuf.baseAddress, recordCount, &errorMsg)
        }
      }
    }
  }

  public func search(
    embedding: [Float],
    limit: Int,
    minSimilarity: Float,
    metadataFilter: [String: String]?
  ) throws -> [RetrievalRecord] {
    let queryCount = try Self.cCount(embedding.count)
    let cLimit = try Self.cCount(limit)
    let filter = metadataFilter ?? [:]
    let filterCount = try Self.cCount(filter.count)
    let arena = CRecordArena()
    defer { arena.deallocateAll() }

    let cFilter: [MpKeyValuePair] = filter.map { key, value in
      MpKeyValuePair(key: arena.cString(key), value: arena.cString(value))
    }
    let cFilterPtr = arena.copyBuffer(cFilter)

    var cResult = MpVectorStoreRecordsResult()
    defer { MpSqliteVectorStoreCloseRecordsResult(&cResult) }

    try embedding.withUnsafeBufferPointer { queryBuf in
      try lock.withLock {
        let store = try ensureStoreUnlocked()
        try MpStatus.check(defaultMessage: "Failed to search vector store.") { errorMsg in
          MpSqliteVectorStoreSearch(
            store,
            queryBuf.baseAddress,
            queryCount,
            cLimit,
            minSimilarity,
            cFilterPtr,
            filterCount,
            &cResult,
            &errorMsg
          )
        }
      }
    }
    return RetrievalRecord.records(from: cResult)
  }

  public func records(ids: [String]) throws -> [RetrievalRecord] {
    guard !ids.isEmpty else { return [] }
    let idsCount = try Self.cCount(ids.count)
    let arena = CRecordArena()
    defer { arena.deallocateAll() }

    let cIdPtrs: [UnsafePointer<CChar>?] = ids.map { Optional(arena.cString($0)) }
    let idsBase = arena.copyBuffer(cIdPtrs)

    var cResult = MpVectorStoreRecordsResult()
    defer { MpSqliteVectorStoreCloseRecordsResult(&cResult) }

    try lock.withLock {
      let store = try ensureStoreUnlocked()
      try MpStatus.check(defaultMessage: "Failed to fetch records.") { errorMsg in
        MpSqliteVectorStoreGetRecords(store, idsBase, idsCount, &cResult, &errorMsg)
      }
    }
    return RetrievalRecord.records(from: cResult)
  }

  public func recordIdentifiers() throws -> [String] {
    var cIds = MpRecordIdsResult()
    defer { MpSemanticRetrieverCloseRecordIdsResult(&cIds) }

    try lock.withLock {
      let store = try ensureStoreUnlocked()
      try MpStatus.check(defaultMessage: "Failed to fetch record identifiers.") { errorMsg in
        MpSqliteVectorStoreGetAllRecordIds(store, &cIds, &errorMsg)
      }
    }
    return Array(cArray: cIds.ids, count: cIds.ids_count) { idPtr in
      idPtr.map { String(cString: $0) } ?? ""
    }
  }

  public func delete(ids: [String]) throws {
    guard !ids.isEmpty else { return }
    let idsCount = try Self.cCount(ids.count)
    let arena = CRecordArena()
    defer { arena.deallocateAll() }

    let cIdPtrs: [UnsafePointer<CChar>?] = ids.map { Optional(arena.cString($0)) }
    let idsBase = arena.copyBuffer(cIdPtrs)

    try lock.withLock {
      let store = try ensureStoreUnlocked()
      try MpStatus.check(defaultMessage: "Failed to delete records by ID.") { errorMsg in
        MpSqliteVectorStoreDeleteByIds(store, idsBase, idsCount, &errorMsg)
      }
    }
  }

  public func delete(metadataFilter: [String: String]) throws {
    guard !metadataFilter.isEmpty else {
      throw TasksError.invalidArgument("metadataFilter cannot be empty.")
    }
    let filterCount = try Self.cCount(metadataFilter.count)
    let arena = CRecordArena()
    defer { arena.deallocateAll() }

    let cFilter: [MpKeyValuePair] = metadataFilter.map { key, value in
      MpKeyValuePair(key: arena.cString(key), value: arena.cString(value))
    }
    let cFilterPtr = arena.copyBuffer(cFilter)

    try lock.withLock {
      let store = try ensureStoreUnlocked()
      try MpStatus.check(defaultMessage: "Failed to delete records by metadata.") { errorMsg in
        MpSqliteVectorStoreDeleteByMetadata(store, cFilterPtr, filterCount, &errorMsg)
      }
    }
  }

  public func deleteAllRecords() throws {
    try lock.withLock {
      let store = try ensureStoreUnlocked()
      try MpStatus.check(defaultMessage: "Failed to delete all records.") { errorMsg in
        MpSqliteVectorStoreDeleteAll(store, &errorMsg)
      }
    }
  }
}

/// Scoped arena owning temporary C string and POD buffers across a single FFI call.
private final class CRecordArena {
  private var deallocators: [() -> Void] = []

  func cString(_ string: String) -> UnsafePointer<CChar> {
    let utf8 = string.utf8CString
    let ptr = UnsafeMutablePointer<CChar>.allocate(capacity: utf8.count)
    _ = UnsafeMutableBufferPointer(start: ptr, count: utf8.count).initialize(from: utf8)
    deallocators.append { ptr.deallocate() }
    return UnsafePointer(ptr)
  }

  func copyBuffer<Element>(_ elements: [Element]) -> UnsafePointer<Element>? {
    guard !elements.isEmpty else { return nil }
    let ptr = UnsafeMutablePointer<Element>.allocate(capacity: elements.count)
    _ = UnsafeMutableBufferPointer(start: ptr, count: elements.count).initialize(from: elements)
    deallocators.append { ptr.deallocate() }
    return UnsafePointer(ptr)
  }

  func deallocateAll() {
    for deallocate in deallocators {
      deallocate()
    }
    deallocators.removeAll()
  }
}
