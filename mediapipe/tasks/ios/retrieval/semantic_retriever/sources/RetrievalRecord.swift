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
internal import third_party_mediapipe_tasks_c_retrieval_semantic_retriever_semantic_retriever

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksComponents
  import MediaPipeTasksCore
#endif

/// A stored entry persisted in or returned from a `VectorStore`.
public struct RetrievalRecord: Sendable, Equatable, Hashable {
  /// Unique identifier of this entry within the store.
  public let id: String

  /// Ordered multimodal segments associated with this entry.
  public let content: [TaskPart]

  /// Dense embedding vector persisted for similarity search.
  public let embeddings: [Float]

  /// Cosine similarity score populated during nearest-neighbor search, if computed by the store.
  public let similarityScore: Float?

  /// Arbitrary key-value attributes used for scoped filtering and caller context.
  public let metadata: [String: String]

  /// Identifier of the parent document when this entry represents a sub-document chunk.
  public let parentId: String?

  /// Identifiers of child chunk entries belonging to this parent document, or empty when
  /// unchunked.
  public let childIds: [String]

  /// Creates a vector store record.
  public init(
    id: String,
    content: [TaskPart],
    embeddings: [Float],
    similarityScore: Float? = nil,
    metadata: [String: String] = [:],
    parentId: String? = nil,
    childIds: [String] = []
  ) {
    self.id = id
    self.content = content
    self.embeddings = embeddings
    self.similarityScore = similarityScore
    self.metadata = metadata
    self.parentId = parentId
    self.childIds = childIds
  }

  init(cRecord: MpVectorStoreRecord) {
    self.id = cRecord.id.map { String(cString: $0) } ?? ""
    self.content = Array(cArray: cRecord.parts, count: cRecord.parts_count) { cPart in
      switch cPart.kind {
      case kMpTaskPartImage:
        let filePath = cPart.image_part.file_path.map { String(cString: $0) }
        return .image(ImagePart(filePath: filePath))
      case kMpTaskPartAudio:
        let filePath = cPart.audio_part.audio_path.map { String(cString: $0) }
        return .audio(AudioPart(filePath: filePath))
      default:
        let text = cPart.text_part.text.map { String(cString: $0) } ?? ""
        return .text(TextPart(text: text))
      }
    }
    self.embeddings = Array(cArray: cRecord.embeddings, count: cRecord.embeddings_count) { $0 }
    self.similarityScore = cRecord.has_similarity_score ? cRecord.similarity_score : nil
    let pairs = Array(cArray: cRecord.metadata, count: cRecord.metadata_count) { pair in
      (pair.key.map { String(cString: $0) } ?? "", pair.value.map { String(cString: $0) } ?? "")
    }
    self.metadata = Dictionary(pairs) { _, last in last }
    self.parentId = cRecord.parent_id.map { String(cString: $0) }
    self.childIds = Array(cArray: cRecord.child_ids, count: cRecord.child_ids_count) { ptr in
      ptr.map { String(cString: $0) } ?? ""
    }
  }

  static func records(from cResult: MpVectorStoreRecordsResult) -> [RetrievalRecord] {
    Array(cArray: cResult.records, count: cResult.records_count, transform: RetrievalRecord.init)
  }
}
