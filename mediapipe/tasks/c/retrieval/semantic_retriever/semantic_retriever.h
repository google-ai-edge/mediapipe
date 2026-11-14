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

#ifndef MEDIAPIPE_TASKS_C_RETRIEVAL_SEMANTIC_RETRIEVER_SEMANTIC_RETRIEVER_H_
#define MEDIAPIPE_TASKS_C_RETRIEVAL_SEMANTIC_RETRIEVER_SEMANTIC_RETRIEVER_H_

#include <stdbool.h>
#include <stdint.h>

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/retrieval/universal_embedder/universal_embedder.h"

#ifdef __cplusplus
#include "mediapipe/tasks/c/text/text_embedder/text_embedder.h"
#include "mediapipe/tasks/c/vision/image_embedder/image_embedder.h"
#endif  // __cplusplus

#ifndef MP_EXPORT
#if defined(_MSC_VER)
#define MP_EXPORT __declspec(dllexport)
#else
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // _MSC_VER
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MpSemanticRetrieverInternal* MpSemanticRetrieverPtr;
typedef struct MpSqliteVectorStoreInternal* MpSqliteVectorStorePtr;

#ifndef __cplusplus
// Opaque embedder handles, declared here so that C clients do not need the
// text and image embedder headers.
typedef struct MpTextEmbedderInternal* MpTextEmbedderPtr;
typedef struct MpImageEmbedderInternal* MpImageEmbedderPtr;
#endif  // __cplusplus

// A simple key-value string pair representation for metadata.
typedef struct MpKeyValuePair {
  const char* key;
  const char* value;
} MpKeyValuePair;

// Represents a retrieved record in the C API.
typedef struct MpRetrievalRecord {
  char* id;
  char* text;
  float score;
  struct MpKeyValuePair* metadata;
  int metadata_count;
} MpRetrievalRecord;

// Represents a list of retrieved records.
typedef struct MpRetrievalResult {
  struct MpRetrievalRecord* records;
  uint32_t records_count;
} MpRetrievalResult;

// Represents the kind of content inside a multimodal query part.
typedef enum MpTaskPartKind {
  kMpTaskPartText = 0,
  kMpTaskPartImage = 1,
  kMpTaskPartAudio = 2
} MpTaskPartKind;

typedef struct MpTextPart {
  const char* text;
} MpTextPart;

typedef struct MpImagePart {
  const uint8_t* image_bytes;
  int image_bytes_size;
  const char* file_path;
} MpImagePart;

typedef struct MpAudioPart {
  const float* audio_data;
  int audio_data_size;
  const char* audio_path;
} MpAudioPart;

// A C-compatible representation of core::TaskPart.
typedef struct MpTaskPart {
  enum MpTaskPartKind kind;
  struct MpTextPart text_part;
  struct MpImagePart image_part;
  struct MpAudioPart audio_part;
} MpTaskPart;

// Mode for text chunking.
typedef enum MpChunkingMode {
  kMpChunkingModeCharacter = 0,
  kMpChunkingModeWord = 1,
} MpChunkingMode;

// The options for configuring a MediaPipe SemanticRetriever task.
typedef struct MpSemanticRetrieverOptions {
  // Base options for configuring MediaPipe Tasks, such as telemetry options.
  struct MpBaseOptions base_options;

  // Path to the SQLite database file. If empty, an in-memory database is used.
  const char* database_path;

  // Dimension of the embedding vector.
  int embedding_dimension;

  // The universal embedder ptr to use for generating embeddings.
  MpUniversalEmbedderPtr embedder;

  // Optional text embedder ptr to use for generating embeddings.
  MpTextEmbedderPtr text_embedder;

  // Optional image embedder ptr to use for generating embeddings.
  MpImageEmbedderPtr image_embedder;

  // The size of each text chunk (default 512).
  int chunk_size;

  // The overlap between consecutive chunks (default 100).
  int chunk_overlap;

  // Mode for text chunking of the default chunker (default character-based).
  enum MpChunkingMode chunking_mode;
} MpSemanticRetrieverOptions;

// Represents a list of record IDs.
typedef struct MpRecordIdsResult {
  char** ids;
  uint32_t ids_count;
} MpRecordIdsResult;

// Represents a single record persisted in or read from `MpSqliteVectorStore`.
// Parent container records (`child_ids_count > 0`) may omit `embeddings`
// (`embeddings == NULL && embeddings_count == 0`).
typedef struct MpVectorStoreRecord {
  const char* id;
  const struct MpTaskPart* parts;
  int parts_count;
  const float* embeddings;
  int embeddings_count;
  bool has_similarity_score;
  float similarity_score;
  const struct MpKeyValuePair* metadata;
  int metadata_count;
  const char* parent_id;
  const char* const* child_ids;
  int child_ids_count;
} MpVectorStoreRecord;

// Represents a list of vector store records.
typedef struct MpVectorStoreRecordsResult {
  struct MpVectorStoreRecord* records;
  uint32_t records_count;
} MpVectorStoreRecordsResult;

// Represents a list of text chunks produced by `MpTextChunkerChunk`.
typedef struct MpTextChunksResult {
  char** chunks;
  uint32_t chunks_count;
} MpTextChunksResult;

// Creates a SemanticRetriever from the provided `options`.
// If successful, returns `kMpOk` and sets `*retriever` to the new
// `MpSemanticRetrieverPtr`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus
MpSemanticRetrieverCreate(struct MpSemanticRetrieverOptions* options,
                          MpSemanticRetrieverPtr* retriever, char** error_msg);

// Inserts a document with specified `id`, `text`, and optional key-value
// `metadata` pairs. If successful, returns `kMpOk`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpSemanticRetrieverInsertDocument(
    MpSemanticRetrieverPtr retriever, const char* id, const char* text,
    struct MpKeyValuePair* metadata, int metadata_count, char** error_msg);

// Inserts an image with specified `id`, optional raw `image_bytes`, and
// standard `file_path`. If raw `image_bytes` are not provided, they are loaded
// and decoded from the `file_path` on-demand. If successful, returns `kMpOk`.
MP_EXPORT MpStatus MpSemanticRetrieverInsertImage(
    MpSemanticRetrieverPtr retriever, const char* id,
    const uint8_t* image_bytes, int image_bytes_size, const char* file_path,
    struct MpKeyValuePair* metadata, int metadata_count, char** error_msg);

// Inserts audio data with specified `id`, optional float `audio_data`, and
// `audio_path`. If raw `audio_data` are not provided, they are loaded and
// decoded from the `audio_path` on-demand. If successful, returns `kMpOk`.
MP_EXPORT MpStatus MpSemanticRetrieverInsertAudio(
    MpSemanticRetrieverPtr retriever, const char* id, const float* audio_data,
    int audio_data_size, const char* audio_path,
    struct MpKeyValuePair* metadata, int metadata_count, char** error_msg);

// Inserts multi-modal parts with specified `id` and optional key-value
// `metadata` pairs. If successful, returns `kMpOk`.
MP_EXPORT MpStatus MpSemanticRetrieverInsertContent(
    MpSemanticRetrieverPtr retriever, const char* id,
    const struct MpTaskPart* parts, int parts_count,
    struct MpKeyValuePair* metadata, int metadata_count, char** error_msg);

// Retrieves nearest records for the given query `parts`.
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpRetrievalResult`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpSemanticRetrieverRetrieve(MpSemanticRetrieverPtr retriever,
                                               const struct MpTaskPart* parts,
                                               int parts_count, int limit,
                                               float min_similarity,
                                               MpRetrievalResult* result,
                                               char** error_msg);

// Retrieves nearest records for the given query `parts` filtered by key-value
// `metadata_filter` pairs.
// If successful, returns `kMpOk` and sets `*result` to the new
// `MpRetrievalResult`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpSemanticRetrieverRetrieveWithMetadataFilter(
    MpSemanticRetrieverPtr retriever, const struct MpTaskPart* parts,
    int parts_count, int limit, const struct MpKeyValuePair* metadata_filter,
    int metadata_filter_count, float min_similarity, MpRetrievalResult* result,
    char** error_msg);

// Deletes a document with the given `id` from the retriever.
// If successful, returns `kMpOk`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpSemanticRetrieverDelete(MpSemanticRetrieverPtr retriever,
                                             const char* id, char** error_msg);

// Deletes documents matching the given key-value `metadata_filter` pairs from
// the retriever.
// If successful, returns `kMpOk`.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpSemanticRetrieverDeleteWithMetadataFilter(
    MpSemanticRetrieverPtr retriever,
    const struct MpKeyValuePair* metadata_filter, int metadata_filter_count,
    char** error_msg);

// Returns all record IDs currently stored in the database.
// The caller is responsible for freeing each returned string and the array
// itself using MpSemanticRetrieverCloseRecordIdsResult.
MP_EXPORT MpStatus MpSemanticRetrieverGetAllRecordIds(
    MpSemanticRetrieverPtr retriever, struct MpRecordIdsResult* result,
    char** error_msg);

// Frees the memory allocated inside an MpRecordIdsResult.
MP_EXPORT void MpSemanticRetrieverCloseRecordIdsResult(
    struct MpRecordIdsResult* result);

// Deletes all records from the database.
MP_EXPORT MpStatus MpSemanticRetrieverDeleteAll(
    MpSemanticRetrieverPtr retriever, char** error_msg);

// Frees the memory allocated inside an MpRetrievalResult result. Does not
// free the result pointer itself.
MP_EXPORT void MpSemanticRetrieverCloseResult(MpRetrievalResult* result);

// Shuts down the SemanticRetriever. Frees all memory.
//
// To obtain a detailed error, `error_msg` must be non-null pointer to a
// `char*`, which will be populated with a newly-allocated error message upon
// failure. It's the caller responsibility to free the error message with
// `MpErrorFree()`.
MP_EXPORT MpStatus MpSemanticRetrieverClose(MpSemanticRetrieverPtr retriever,
                                            char** error_msg);

// Creates a SQLite-backed vector store at `database_path` (or in-memory when
// `database_path` is null, empty, or `":memory:"`) for vectors of size
// `embedding_dimension`.
MP_EXPORT MpStatus MpSqliteVectorStoreCreate(const char* database_path,
                                             int embedding_dimension,
                                             MpSqliteVectorStorePtr* store,
                                             char** error_msg);

// Inserts or replaces `records_count` records in `store`.
//
// Validates all input records upfront before mutating `store`, then deletes any
// superseded rows with matching IDs and stale child chunk rows before inserting
// the batch. Because deletion and batch insertion execute in separate SQLite
// statements/transactions, replacing existing records is not atomic across a
// subsequent SQLite I/O failure.
MP_EXPORT MpStatus MpSqliteVectorStoreUpsert(
    MpSqliteVectorStorePtr store, const struct MpVectorStoreRecord* records,
    int records_count, char** error_msg);

// Searches `store` for up to `limit` records whose cosine similarity to
// `query_embedding` is at least `min_similarity_score`, optionally restricted
// to records matching all `metadata_filter` pairs. Free `result` with
// `MpSqliteVectorStoreCloseRecordsResult`.
MP_EXPORT MpStatus MpSqliteVectorStoreSearch(
    MpSqliteVectorStorePtr store, const float* query_embedding,
    int query_embedding_size, int limit, float min_similarity_score,
    const struct MpKeyValuePair* metadata_filter, int metadata_filter_count,
    struct MpVectorStoreRecordsResult* result, char** error_msg);

// Fetches all stored records whose identifiers appear in `ids`. Free `result`
// with `MpSqliteVectorStoreCloseRecordsResult`.
MP_EXPORT MpStatus MpSqliteVectorStoreGetRecords(
    MpSqliteVectorStorePtr store, const char* const* ids, int ids_count,
    struct MpVectorStoreRecordsResult* result, char** error_msg);

// Returns all top-level record identifiers in `store`. Free `result` with
// `MpSemanticRetrieverCloseRecordIdsResult`.
MP_EXPORT MpStatus MpSqliteVectorStoreGetAllRecordIds(
    MpSqliteVectorStorePtr store, struct MpRecordIdsResult* result,
    char** error_msg);

// Deletes all records matching `ids` (including child chunk/part rows).
MP_EXPORT MpStatus MpSqliteVectorStoreDeleteByIds(MpSqliteVectorStorePtr store,
                                                  const char* const* ids,
                                                  int ids_count,
                                                  char** error_msg);

// Deletes all records matching all `metadata_filter` pairs (including child
// chunk/part rows).
MP_EXPORT MpStatus MpSqliteVectorStoreDeleteByMetadata(
    MpSqliteVectorStorePtr store, const struct MpKeyValuePair* metadata_filter,
    int metadata_filter_count, char** error_msg);

// Deletes every record in `store`.
MP_EXPORT MpStatus MpSqliteVectorStoreDeleteAll(MpSqliteVectorStorePtr store,
                                                char** error_msg);

// Frees memory owned inside `result` without freeing `result` itself.
MP_EXPORT void MpSqliteVectorStoreCloseRecordsResult(
    struct MpVectorStoreRecordsResult* result);

// Closes `store` and releases its underlying SQLite connection.
MP_EXPORT MpStatus MpSqliteVectorStoreClose(MpSqliteVectorStorePtr store,
                                            char** error_msg);

// Splits UTF-8 `text` into chunks according to `mode`. Free `result` with
// `MpTextChunkerCloseResult`.
MP_EXPORT MpStatus MpTextChunkerChunk(const char* text, int chunk_size,
                                      int chunk_overlap, int word_threshold,
                                      enum MpChunkingMode mode,
                                      struct MpTextChunksResult* result,
                                      char** error_msg);

// Frees memory owned inside `result` without freeing `result` itself.
MP_EXPORT void MpTextChunkerCloseResult(struct MpTextChunksResult* result);

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_RETRIEVAL_SEMANTIC_RETRIEVER_SEMANTIC_RETRIEVER_H_
