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

/// Granularity used to partition long documents before embedding.
public enum ChunkingMode: Int, Sendable, Equatable, Hashable, CaseIterable {
  /// Splits text into fixed-length character sliding windows.
  case character = 0

  /// Splits text along paragraph and sentence boundaries bounded by word counts.
  case word = 1

  var cChunkingMode: MpChunkingMode {
    self == .word ? kMpChunkingModeWord : kMpChunkingModeCharacter
  }
}

/// Splits long UTF-8 documents into smaller segments suitable for embedding.
public protocol TextChunker {
  /// Partitions `text` into ordered segments.
  ///
  /// - Parameter text: Source document text to partition.
  /// - Returns: Ordered text chunks, or an empty array when `text` contains no non-whitespace text.
  func chunk(text: String) throws -> [String]
}

/// Built-in text chunker supporting character sliding windows and layout-preserving word windows.
public struct DefaultTextChunker: Sendable, Equatable, Hashable {
  /// Target window size in characters (`.character`) or minimum split boundary in words (`.word`).
  public let chunkSize: Int

  /// Window overlap in characters (`.character`) or word delta when `wordThreshold` is omitted.
  public let chunkOverlap: Int

  /// Maximum word count per chunk in `.word` mode (`0` in `.character` mode).
  public let wordThreshold: Int

  /// Partitioning granularity.
  public let mode: ChunkingMode

  /// Creates a text chunker after validating window bounds for `mode`.
  ///
  /// When `mode == .word` and `wordThreshold` is `nil`, `chunkSize` is used as `wordThreshold` and
  /// `chunkSize - chunkOverlap` becomes the minimum sentence-split word count.
  ///
  /// - Throws: `TasksError` with `.invalidArgument` if the window bounds are invalid.
  public init(
    chunkSize: Int = 500,
    chunkOverlap: Int = 100,
    wordThreshold: Int? = nil,
    mode: ChunkingMode = .character
  ) throws {
    let resolvedChunkSize: Int
    let resolvedWordThreshold: Int
    if mode == .word, wordThreshold == nil {
      resolvedWordThreshold = chunkSize
      resolvedChunkSize = chunkSize - chunkOverlap
    } else {
      resolvedChunkSize = chunkSize
      resolvedWordThreshold = wordThreshold ?? 0
    }

    guard resolvedChunkSize > 0, Int32(exactly: resolvedChunkSize) != nil else {
      throw TasksError.invalidArgument("Chunk size must be positive.")
    }

    switch mode {
    case .character:
      guard chunkOverlap >= 0, chunkOverlap < resolvedChunkSize,
        Int32(exactly: chunkOverlap) != nil
      else {
        throw TasksError.invalidArgument(
          "Chunk overlap must be non-negative and less than chunk size."
        )
      }
    case .word:
      guard resolvedWordThreshold > 0, Int32(exactly: resolvedWordThreshold) != nil else {
        throw TasksError.invalidArgument("Word threshold must be positive.")
      }
      guard resolvedChunkSize <= resolvedWordThreshold,
        chunkOverlap >= 0, Int32(exactly: chunkOverlap) != nil
      else {
        throw TasksError.invalidArgument(
          "Chunk size (word chunk size) must be less than or equal to word threshold."
        )
      }
    }

    self.chunkSize = resolvedChunkSize
    self.chunkOverlap = chunkOverlap
    self.wordThreshold = resolvedWordThreshold
    self.mode = mode
  }
}

extension DefaultTextChunker: TextChunker {
  public func chunk(text: String) throws -> [String] {
    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      return []
    }
    var cResult = MpTextChunksResult()
    defer { MpTextChunkerCloseResult(&cResult) }
    try text.withCString { utf8Ptr in
      try MpStatus.check(defaultMessage: "Failed to chunk text.") { errorMsg in
        MpTextChunkerChunk(
          utf8Ptr,
          Int32(chunkSize),
          Int32(chunkOverlap),
          Int32(wordThreshold),
          mode.cChunkingMode,
          &cResult,
          &errorMsg
        )
      }
    }
    return Array(cArray: cResult.chunks, count: cResult.chunks_count) { chunkPtr in
      chunkPtr.map { String(cString: $0) } ?? ""
    }
  }
}
