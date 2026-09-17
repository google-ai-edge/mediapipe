/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_RETRIEVAL_SEMANTIC_RETRIEVER_SEMANTIC_RETRIEVER_H_
#define MEDIAPIPE_TASKS_CC_RETRIEVAL_SEMANTIC_RETRIEVER_SEMANTIC_RETRIEVER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/core/embedding_provider.h"
#include "mediapipe/tasks/cc/core/logging/tasks_logger.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/memory_store.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/text_chunker.h"

namespace mediapipe::tasks::retrieval {

// Components required to construct a SemanticRetriever instance.
struct SemanticRetrieverComponents {
  // The data store used to save, retrieve, and delete memory records. Must not
  // be null.
  std::unique_ptr<MemoryStore> memory_store;

  // The provider used to extract embedding vectors from task content. Must not
  // be null.
  std::unique_ptr<core::EmbeddingProvider> embedding_provider;

  // Optional chunker used to split long text documents. If null, a default
  // character-based text chunker will be used.
  std::unique_ptr<TextChunker> text_chunker;
};

class SemanticRetriever {
 public:
  // Creates and initializes a SemanticRetriever instance.
  //
  // @param components Struct containing dependencies like MemoryStore,
  // EmbeddingProvider, and optional TextChunker.
  // @return A unique_ptr to the created SemanticRetriever, or a non-OK status.
  static absl::StatusOr<std::unique_ptr<SemanticRetriever>> Create(
      SemanticRetrieverComponents components);

  // Destructor that releases dependencies and logs session termination.
  ~SemanticRetriever();

  // Inserts an text document into the memory store.
  //
  // If the document is large, it may be chunked into smaller passages using the
  // configured TextChunker. The text and its computed embeddings are stored in
  // the memory store under the specified ID.
  //
  // @param id The unique identifier for the document.
  // @param text The full raw text of the document.
  // @param metadata Optional key-value metadata pairs to associate with the
  // record.
  // @return OK status on success, or an error status on failure.
  absl::Status InsertDocument(
      absl::string_view id, absl::string_view text,
      const absl::flat_hash_map<std::string, std::string>& metadata = {});

  // Inserts multi-modal content parts under a single ID.
  //
  // Iterates over a list of mixed parts (text, image, and audio). Any text part
  // that exceeds chunk bounds is automatically chunked and stored as a
  // parent-child relationship, matching iOS and Java features.
  //
  // @param id The unique identifier for the content record.
  // @param parts Vector of task parts (text, image, audio) representing the
  // content.
  // @param metadata Optional key-value metadata pairs to associate with the
  // record.
  // @return OK status on success, or an error status on failure.
  absl::Status InsertContent(
      absl::string_view id, const std::vector<core::TaskPart>& parts,
      const absl::flat_hash_map<std::string, std::string>& metadata = {});

  // Inserts an image into the memory store.
  //
  // Loads and decodes the image file from the specified path, computes its
  // embedding, and stores the image path and embedding under the specified ID.
  //
  // @param id The unique identifier for the image.
  // @param file_path Path to the image file.
  // @param metadata Optional key-value metadata pairs to associate with the
  // record.
  // @return OK status on success, or an error status on failure.
  absl::Status InsertImage(
      absl::string_view id, absl::string_view file_path,
      const absl::flat_hash_map<std::string, std::string>& metadata = {});

  // Inserts an image into the memory store using raw encoded image bytes
  // directly.
  //
  // Computes the embedding from the raw image bytes (e.g., JPEG, PNG) in
  // memory. The raw bytes are discarded, and only the computed embedding and
  // the provided `file_path` are stored in the database.
  //
  // @param id The unique identifier for the image record.
  // @param image_bytes Raw encoded image bytes (e.g. file content of a PNG or
  // JPEG).
  // @param file_path Path or URI for the image to be stored in the database.
  // @param metadata Optional key-value metadata pairs to associate with the
  // record.
  // @return OK status on success, or an error status on failure.
  absl::Status InsertImage(
      absl::string_view id, absl::string_view image_bytes,
      absl::string_view file_path,
      const absl::flat_hash_map<std::string, std::string>& metadata = {});

  // Inserts an audio file into the memory store by decoding a WAV file.
  //
  // Loads the WAV file from the specified path, decodes it into mono PCM float
  // values, computes its embedding, and stores it under the specified ID.
  //
  // @param id The unique identifier for the audio record.
  // @param file_path Path to the 16-bit PCM mono WAV file.
  // @param metadata Optional key-value metadata pairs to associate with the
  // record.
  // @return OK status on success, or an error status on failure.
  absl::Status InsertAudio(
      absl::string_view id, absl::string_view file_path,
      const absl::flat_hash_map<std::string, std::string>& metadata = {});

  // Inserts pre-decoded raw PCM audio data directly into the memory store.
  //
  // Computes the embedding from the raw mono PCM buffer in memory. The raw
  // audio data is discarded, and only the computed embedding and the provided
  // `file_path` are stored in the database.
  //
  // @param id The unique identifier for the audio record.
  // @param audio_data Raw PCM mono audio samples normalized in range
  // [-1.0, 1.0].
  // @param file_path Path or URI for the audio to be stored in the database.
  // @param metadata Optional key-value metadata pairs to associate with the
  // record.
  // @return OK status on success, or an error status on failure.
  absl::Status InsertAudio(
      absl::string_view id, const std::vector<float>& audio_data,
      absl::string_view file_path,
      const absl::flat_hash_map<std::string, std::string>& metadata = {});

  // Retrieves top-k nearest matching records based on a vector of task parts
  // (multimodal).
  //
  // Computes the query embedding, searches the memory store for nearest
  // records, and returns parent records for any matched chunks or original
  // standalone records.
  //
  // @param query_content Vector of input task parts (text, image, audio)
  // representing the query.
  // @param limit The maximum number of nearest records to return.
  // @param metadata_filter Key-value pairs used to filter candidate records.
  // @param min_similarity Minimum similarity score threshold for matched
  // results.
  // @return A vector of matched MemoryRecord objects, or an error status.
  absl::StatusOr<std::vector<MemoryRecord>> Retrieve(
      const std::vector<core::TaskPart>& query_content, int limit,
      const absl::flat_hash_map<std::string, std::string>& metadata_filter,
      float min_similarity = 0.5f);

  absl::StatusOr<std::vector<MemoryRecord>> Retrieve(
      const std::vector<core::TaskPart>& query_content, int limit,
      float min_similarity = 0.5f);

  // Retrieves top-k nearest matching records based on a text query string with
  // optional metadata filtering.
  //
  // @param query The text query string.
  // @param limit The maximum number of nearest records to return.
  // @param metadata_filter Key-value pairs used to filter candidate records.
  // @param min_similarity Minimum similarity score threshold for matched
  // results.
  // @return A vector of matched MemoryRecord objects, or an error status.
  absl::StatusOr<std::vector<MemoryRecord>> Retrieve(
      absl::string_view query, int limit,
      const absl::flat_hash_map<std::string, std::string>& metadata_filter,
      float min_similarity = 0.5f);

  absl::StatusOr<std::vector<MemoryRecord>> Retrieve(
      absl::string_view query, int limit, float min_similarity = 0.5f);

  // Deletes a document and all of its associated chunks from the memory store.
  //
  // @param id The unique identifier of the document/record to delete.
  // @return OK status on success, or an error status on failure.
  absl::Status Delete(absl::string_view id);

  // Deletes documents and associated chunks matching the metadata filter.
  //
  // @param metadata_filter Key-value pairs used to match records for deletion.
  // @return OK status on success, or an error status on failure.
  absl::Status Delete(
      const absl::flat_hash_map<std::string, std::string>& metadata_filter);

  // Retrieves all unique record_ids stored in the retriever.
  //
  // @return A vector of unique record IDs, or an error status.
  absl::StatusOr<std::vector<std::string>> GetAllRecordIds();

  // Deletes all documents and associated records from the memory store.
  //
  // @return OK status on success, or an error status on failure.
  absl::Status DeleteAll();

 private:
  explicit SemanticRetriever(
      std::unique_ptr<MemoryStore> memory_store,
      std::unique_ptr<core::EmbeddingProvider> embedding_provider,
      std::unique_ptr<TextChunker> text_chunker,
      std::unique_ptr<core::logging::TasksLogger> tasks_logger)
      : memory_store_(std::move(memory_store)),
        embedding_provider_(std::move(embedding_provider)),
        text_chunker_(std::move(text_chunker)),
        tasks_logger_(std::move(tasks_logger)) {}

  std::unique_ptr<MemoryStore> memory_store_;
  std::unique_ptr<core::EmbeddingProvider> embedding_provider_;
  std::unique_ptr<TextChunker> text_chunker_;
  std::unique_ptr<core::logging::TasksLogger> tasks_logger_;
  int64_t tasks_logger_timestamp_ = 0;
};

}  // namespace mediapipe::tasks::retrieval

#endif  // MEDIAPIPE_TASKS_CC_RETRIEVAL_SEMANTIC_RETRIEVER_SEMANTIC_RETRIEVER_H_
