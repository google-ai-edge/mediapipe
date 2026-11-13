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
package import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_text_text_embedder_text_embedder

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
#endif

/// Task intent used to format input text for prompt-aware embedding models.
public enum EmbeddingType: Int, Sendable, Equatable, Hashable, CaseIterable {
  /// Formats text for a search or retrieval query.
  case retrievalQuery = 1
  /// Formats text as a retrievable document.
  case retrievalDocument = 2
  /// Formats text for semantic similarity comparison.
  case semanticSimilarity = 3
  /// Formats text for downstream classification.
  case classification = 4
  /// Formats text for clustering.
  case clustering = 5
  /// Formats text for question answering.
  case questionAnswering = 6
  /// Formats text for fact verification.
  case factChecking = 7
  /// Formats text for code retrieval.
  case codeRetrieval = 8

  package var cEmbeddingType: MpTextEmbedderEmbeddingType {
    switch self {
    case .retrievalQuery: MP_TEXT_EMBEDDER_EMBEDDING_TYPE_RETRIEVAL_QUERY
    case .retrievalDocument: MP_TEXT_EMBEDDER_EMBEDDING_TYPE_RETRIEVAL_DOCUMENT
    case .semanticSimilarity: MP_TEXT_EMBEDDER_EMBEDDING_TYPE_SEMANTIC_SIMILARITY
    case .classification: MP_TEXT_EMBEDDER_EMBEDDING_TYPE_CLASSIFICATION
    case .clustering: MP_TEXT_EMBEDDER_EMBEDDING_TYPE_CLUSTERING
    case .questionAnswering: MP_TEXT_EMBEDDER_EMBEDDING_TYPE_QUESTION_ANSWERING
    case .factChecking: MP_TEXT_EMBEDDER_EMBEDDING_TYPE_FACT_CHECKING
    case .codeRetrieval: MP_TEXT_EMBEDDER_EMBEDDING_TYPE_CODE_RETRIEVAL
    }
  }
}

/// Role of input text within a prompt-formatted embedding task.
public enum TextRole: Int, Sendable, Equatable, Hashable, CaseIterable {
  /// Indicates a search or retrieval query.
  case query = 1
  /// Indicates an indexed or stored document.
  case document = 2

  package var cRole: MpTextEmbedderRole {
    switch self {
    case .query: MP_TEXT_EMBEDDER_ROLE_QUERY
    case .document: MP_TEXT_EMBEDDER_ROLE_DOCUMENT
    }
  }
}

/// Prompt-formatting instructions for models that distinguish task intent and role (such as Gecko).
public struct TextFormatContext: Sendable, Equatable, Hashable {
  /// Task intent used to format the input text.
  public var embeddingType: EmbeddingType
  /// Optional document title, used when `embeddingType` is `.retrievalDocument`.
  public var title: String?
  /// Role of the input text within the embedding task.
  public var textRole: TextRole

  /// Creates prompt-formatting instructions for text embedding extraction.
  public init(
    embeddingType: EmbeddingType = .retrievalQuery,
    title: String? = nil,
    textRole: TextRole = .query
  ) {
    self.embeddingType = embeddingType
    self.title = title
    self.textRole = textRole
  }

  package func withCFormatContext<Result>(
    _ body: (MpTextEmbedderFormatContext) throws -> Result
  ) rethrows -> Result {
    try title.withCString { titlePtr in
      try body(
        MpTextEmbedderFormatContext(
          task_type: embeddingType.cEmbeddingType,
          title: titlePtr,
          role: textRole.cRole
        )
      )
    }
  }
}

/// Extracts numerical feature embeddings from UTF-8 text using a MediaPipe Tasks model.
///
/// - Note: This class is not thread-safe. Concurrent callers should serialize access (for example,
///   by isolating an instance inside a Swift `actor`).
public final class TextEmbedder {
  private let cEmbedder: MpTextEmbedderPtr

  /// Initializes the MediaPipe text embedder task graph from `options`.
  ///
  /// - Throws: `TasksError` if the model asset cannot be loaded or graph initialization fails.
  public init(options: TextEmbedderOptions) throws {
    self.cEmbedder = try options.withCOptions { cOptions in
      var mutableOptions = cOptions
      var embedderPtr: MpTextEmbedderPtr?
      try MpStatus.check(defaultMessage: "Failed to create TextEmbedder.") { errorMessage in
        MpTextEmbedderCreate(&mutableOptions, &embedderPtr, &errorMessage)
      }
      guard let embedderPtr else {
        throw TasksError.internalError("MpTextEmbedderCreate returned a null pointer.")
      }
      return embedderPtr
    }
  }

  /// Initializes a `TextEmbedder` from a local model file path with default options.
  public convenience init(modelPath: String) throws {
    var options = TextEmbedderOptions()
    options.baseOptions.modelAssetPath = modelPath
    try self.init(options: options)
  }

  deinit {
    _ = MpTextEmbedderClose(cEmbedder, nil)
  }

  /// Extracts feature embeddings for `text` using optional prompt-formatting instructions.
  public func embed(
    text: String,
    textFormatContext: TextFormatContext? = nil
  ) throws -> TextEmbedderResult {
    var cResult = MpTextEmbedderResult()
    defer { MpTextEmbedderCloseResult(&cResult) }

    try text.withCString { utf8Ptr in
      let invokeEmbed: (UnsafePointer<MpTextEmbedderFormatContext>?) throws -> Void = {
        [cEmbedder] cContextPtr in
        try MpStatus.check(defaultMessage: "Failed to embed text.") { errorMessage in
          MpTextEmbedderEmbed(cEmbedder, utf8Ptr, cContextPtr, &cResult, &errorMessage)
        }
      }
      if let textFormatContext {
        try textFormatContext.withCFormatContext { cContext in
          try withUnsafePointer(to: cContext) { try invokeEmbed($0) }
        }
      } else {
        try invokeEmbed(nil)
      }
    }
    return TextEmbedderResult(cResult: cResult)
  }

  /// Computes the cosine similarity between two embeddings in `[-1.0, 1.0]`.
  ///
  /// - Throws: `TasksError` with `.invalidArgument` if the embeddings differ in kind or length,
  ///   are empty, or have a zero L2 norm.
  public static func cosineSimilarity(
    embedding1: Embedding,
    embedding2: Embedding
  ) throws -> Double {
    try CosineSimilarity.compute(between: embedding1, and: embedding2)
  }
}
