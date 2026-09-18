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

#include "mediapipe/tasks/c/retrieval/semantic_retriever/semantic_retriever.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/retrieval/universal_embedder/universal_embedder_internal.h"
#include "mediapipe/tasks/c/text/text_embedder/text_embedder_internal.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/image_embedder/image_embedder_internal.h"
#include "mediapipe/tasks/cc/core/embedding_provider.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/semantic_retriever.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/sqlite_memory_store.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/text_chunker.h"
#include "mediapipe/tasks/cc/retrieval/universal_embedder/universal_embedder.h"
#include "mediapipe/tasks/cc/text/text_embedder/text_embedder.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder.h"

struct MpSemanticRetrieverInternal {
  std::unique_ptr<::mediapipe::tasks::retrieval::SemanticRetriever> instance;
};

namespace mediapipe::tasks::c::retrieval::semantic_retriever {

namespace {

using ::mediapipe::tasks::retrieval::MemoryRecord;
using ::mediapipe::tasks::retrieval::SemanticRetriever;

constexpr int kDefaultChunkSize = 512;
constexpr int kDefaultChunkOverlap = 100;
constexpr int kDefaultEmbeddingDimension = 768;

SemanticRetriever* GetCppRetriever(MpSemanticRetrieverPtr wrapper) {
  ABSL_CHECK(wrapper != nullptr) << "SemanticRetriever is null.";
  return wrapper->instance.get();
}

void CppConvertToRetrievalResult(const std::vector<MemoryRecord>& cpp_records,
                                 MpRetrievalResult* result) {
  result->records_count = cpp_records.size();
  if (cpp_records.empty()) {
    result->records = nullptr;
    return;
  }
  result->records = new MpRetrievalRecord[cpp_records.size()];
  for (size_t i = 0; i < cpp_records.size(); ++i) {
    const auto& cpp_rec = cpp_records[i];
    result->records[i].id = strdup(cpp_rec.record_id().c_str());
    result->records[i].text = strdup(cpp_rec.text().c_str());
    result->records[i].score = cpp_rec.similarity_score();

    // Convert metadata key-value pairs
    if (cpp_rec.has_metadata() &&
        !cpp_rec.metadata().key_value_pairs().empty()) {
      int count = cpp_rec.metadata().key_value_pairs_size();
      result->records[i].metadata_count = count;
      result->records[i].metadata = new MpKeyValuePair[count];
      for (int j = 0; j < count; ++j) {
        const auto& pair = cpp_rec.metadata().key_value_pairs(j);
        result->records[i].metadata[j].key = strdup(pair.key().c_str());
        result->records[i].metadata[j].value = strdup(pair.value().c_str());
      }
    } else {
      result->records[i].metadata = nullptr;
      result->records[i].metadata_count = 0;
    }
  }
}

std::vector<::mediapipe::tasks::core::TaskPart> ConvertQueryParts(
    const struct MpTaskPart* query_parts, int query_parts_count) {
  std::vector<::mediapipe::tasks::core::TaskPart> cpp_parts;
  if (query_parts == nullptr || query_parts_count <= 0) {
    return cpp_parts;
  }
  cpp_parts.reserve(query_parts_count);
  for (int i = 0; i < query_parts_count; ++i) {
    const auto& c_part = query_parts[i];
    if (c_part.kind == kMpTaskPartText) {
      ::mediapipe::tasks::core::TextPart text_part;
      if (c_part.text_part.text != nullptr) {
        text_part.text = c_part.text_part.text;
      }
      cpp_parts.push_back(text_part);
    } else if (c_part.kind == kMpTaskPartImage) {
      ::mediapipe::tasks::core::ImagePart image_part;
      if (c_part.image_part.image_bytes != nullptr &&
          c_part.image_part.image_bytes_size > 0) {
        image_part.image_bytes = std::string(
            reinterpret_cast<const char*>(c_part.image_part.image_bytes),
            c_part.image_part.image_bytes_size);
      }
      if (c_part.image_part.file_path != nullptr) {
        image_part.file_path = c_part.image_part.file_path;
      }
      cpp_parts.push_back(image_part);
    } else if (c_part.kind == kMpTaskPartAudio) {
      ::mediapipe::tasks::core::AudioPart audio_part;
      if (c_part.audio_part.audio_data != nullptr &&
          c_part.audio_part.audio_data_size > 0) {
        audio_part.audio_data.assign(
            c_part.audio_part.audio_data,
            c_part.audio_part.audio_data + c_part.audio_part.audio_data_size);
      }
      if (c_part.audio_part.audio_path != nullptr) {
        audio_part.file_path = c_part.audio_part.audio_path;
      }
      cpp_parts.push_back(audio_part);
    }
  }
  return cpp_parts;
}

}  // namespace

absl::Status CppSemanticRetrieverCreate(
    const MpSemanticRetrieverOptions& options,
    MpSemanticRetrieverPtr* retriever) {
  std::unique_ptr<::mediapipe::tasks::core::EmbeddingProvider>
      embedding_provider;

  if (options.embedder != nullptr) {
    if (options.embedder->instance == nullptr) {
      return absl::InvalidArgumentError(
          "Passed universal embedder wrapper is closed or uninitialized.");
    }
    auto cpp_embedder = options.embedder->instance.get();
    embedding_provider = cpp_embedder->GetProvider();
  } else if (options.text_embedder != nullptr) {
    if (options.text_embedder->instance == nullptr) {
      return absl::InvalidArgumentError(
          "Passed text embedder wrapper is closed or uninitialized.");
    }
    auto cpp_embedder = options.text_embedder->instance.get();
    embedding_provider = cpp_embedder->GetProvider();
  } else if (options.image_embedder != nullptr) {
    if (options.image_embedder->instance == nullptr) {
      return absl::InvalidArgumentError(
          "Passed image embedder wrapper is closed or uninitialized.");
    }
    auto cpp_embedder = options.image_embedder->instance.get();
    embedding_provider = cpp_embedder->GetProvider();
  } else {
    return absl::InvalidArgumentError(
        "At least one embedder backend must be specified in "
        "MpSemanticRetrieverOptions.");
  }

  std::string db_path =
      (options.database_path && options.database_path[0] != '\0')
          ? options.database_path
          : "";
  ABSL_ASSIGN_OR_RETURN(
      auto memory_store,
      ::mediapipe::tasks::retrieval::SqliteMemoryStore::Create(
          db_path,
          (options.embedding_dimension > 0 ? options.embedding_dimension
                                           : kDefaultEmbeddingDimension)));

  ::mediapipe::tasks::retrieval::SemanticRetrieverComponents components;
  components.memory_store = std::move(memory_store);
  components.embedding_provider = std::move(embedding_provider);

  int chunk_size =
      options.chunk_size > 0 ? options.chunk_size : kDefaultChunkSize;
  int chunk_overlap =
      options.chunk_overlap >= 0 ? options.chunk_overlap : kDefaultChunkOverlap;
  ::mediapipe::tasks::retrieval::ChunkingMode cpp_chunking_mode =
      options.chunking_mode == kMpChunkingModeWord
          ? ::mediapipe::tasks::retrieval::ChunkingMode::kWord
          : ::mediapipe::tasks::retrieval::ChunkingMode::kCharacter;
  components.text_chunker =
      std::make_unique<::mediapipe::tasks::retrieval::DefaultTextChunker>(
          chunk_size, chunk_overlap, cpp_chunking_mode);

  auto retriever_or = SemanticRetriever::Create(std::move(components));
  if (!retriever_or.ok()) {
    return retriever_or.status();
  }

  *retriever =
      new MpSemanticRetrieverInternal{.instance = std::move(*retriever_or)};

  return absl::OkStatus();
}

absl::flat_hash_map<std::string, std::string> ConvertMetadataMap(
    const struct MpKeyValuePair* metadata, int metadata_count) {
  absl::flat_hash_map<std::string, std::string> cpp_metadata;
  if (metadata == nullptr || metadata_count <= 0) {
    return cpp_metadata;
  }
  cpp_metadata.reserve(metadata_count);
  for (int i = 0; i < metadata_count; ++i) {
    if (metadata[i].key != nullptr && metadata[i].value != nullptr) {
      cpp_metadata[metadata[i].key] = metadata[i].value;
    }
  }
  return cpp_metadata;
}

absl::Status CppSemanticRetrieverInsertDocument(
    MpSemanticRetrieverPtr retriever, absl::string_view id,
    absl::string_view text, struct MpKeyValuePair* metadata,
    int metadata_count) {
  auto cpp_retriever = GetCppRetriever(retriever);
  return cpp_retriever->InsertDocument(
      id, text, ConvertMetadataMap(metadata, metadata_count));
}

absl::Status CppSemanticRetrieverInsertImage(MpSemanticRetrieverPtr retriever,
                                             absl::string_view id,
                                             absl::string_view image_bytes,
                                             absl::string_view file_path,
                                             struct MpKeyValuePair* metadata,
                                             int metadata_count) {
  auto cpp_retriever = GetCppRetriever(retriever);
  auto converted_metadata = ConvertMetadataMap(metadata, metadata_count);
  if (file_path.empty()) {
    return absl::InvalidArgumentError("file_path must be specified.");
  }
  if (!image_bytes.empty()) {
    return cpp_retriever->InsertImage(id, image_bytes, file_path,
                                      converted_metadata);
  } else {
    return cpp_retriever->InsertImage(id, file_path, converted_metadata);
  }
}

absl::Status CppSemanticRetrieverInsertAudio(
    MpSemanticRetrieverPtr retriever, absl::string_view id,
    const float* audio_data, int audio_data_size, absl::string_view audio_path,
    struct MpKeyValuePair* metadata, int metadata_count) {
  auto cpp_retriever = GetCppRetriever(retriever);
  auto converted_metadata = ConvertMetadataMap(metadata, metadata_count);
  if (audio_path.empty()) {
    return absl::InvalidArgumentError("audio_path must be specified.");
  }
  if (audio_data_size > 0 && audio_data != nullptr) {
    std::vector<float> audio_vector(audio_data, audio_data + audio_data_size);
    return cpp_retriever->InsertAudio(id, audio_vector, audio_path,
                                      converted_metadata);
  } else {
    return cpp_retriever->InsertAudio(id, audio_path, converted_metadata);
  }
}

absl::Status CppSemanticRetrieverInsertContent(MpSemanticRetrieverPtr retriever,
                                               absl::string_view id,
                                               const struct MpTaskPart* parts,
                                               int parts_count,
                                               struct MpKeyValuePair* metadata,
                                               int metadata_count) {
  auto cpp_retriever = GetCppRetriever(retriever);
  auto cpp_parts = ConvertQueryParts(parts, parts_count);
  auto cpp_metadata = ConvertMetadataMap(metadata, metadata_count);
  return cpp_retriever->InsertContent(id, cpp_parts, cpp_metadata);
}

absl::Status CppSemanticRetrieverRetrieve(MpSemanticRetrieverPtr retriever,
                                          const struct MpTaskPart* parts,
                                          int parts_count, int limit,
                                          float min_similarity,
                                          MpRetrievalResult* result) {
  auto cpp_retriever = GetCppRetriever(retriever);
  auto cpp_parts = ConvertQueryParts(parts, parts_count);
  ABSL_ASSIGN_OR_RETURN(
      auto cpp_results,
      cpp_retriever->Retrieve(cpp_parts, limit, min_similarity));
  CppConvertToRetrievalResult(cpp_results, result);
  return absl::OkStatus();
}

absl::Status CppSemanticRetrieverRetrieveWithMetadataFilter(
    MpSemanticRetrieverPtr retriever, const struct MpTaskPart* parts,
    int parts_count, int limit, const struct MpKeyValuePair* metadata_filter,
    int metadata_filter_count, float min_similarity,
    MpRetrievalResult* result) {
  auto cpp_retriever = GetCppRetriever(retriever);
  auto cpp_parts = ConvertQueryParts(parts, parts_count);
  auto cpp_filter = ConvertMetadataMap(metadata_filter, metadata_filter_count);
  ABSL_ASSIGN_OR_RETURN(
      auto cpp_results,
      cpp_retriever->Retrieve(cpp_parts, limit, cpp_filter, min_similarity));
  CppConvertToRetrievalResult(cpp_results, result);
  return absl::OkStatus();
}

absl::Status CppSemanticRetrieverDelete(MpSemanticRetrieverPtr retriever,
                                        absl::string_view id) {
  auto cpp_retriever = GetCppRetriever(retriever);
  return cpp_retriever->Delete(id);
}

absl::Status CppSemanticRetrieverDeleteWithMetadataFilter(
    MpSemanticRetrieverPtr retriever,
    const struct MpKeyValuePair* metadata_filter, int metadata_filter_count) {
  auto cpp_retriever = GetCppRetriever(retriever);
  auto cpp_filter = ConvertMetadataMap(metadata_filter, metadata_filter_count);
  return cpp_retriever->Delete(cpp_filter);
}

absl::Status CppSemanticRetrieverGetAllRecordIds(
    MpSemanticRetrieverPtr retriever, struct MpRecordIdsResult* result) {
  auto cpp_retriever = GetCppRetriever(retriever);
  ABSL_ASSIGN_OR_RETURN(auto cpp_ids, cpp_retriever->GetAllRecordIds());
  result->ids_count = cpp_ids.size();
  if (cpp_ids.empty()) {
    result->ids = nullptr;
    return absl::OkStatus();
  }
  result->ids = new char*[cpp_ids.size()];
  for (size_t i = 0; i < cpp_ids.size(); ++i) {
    result->ids[i] = strdup(cpp_ids[i].c_str());
  }
  return absl::OkStatus();
}

void CppSemanticRetrieverCloseRecordIdsResult(
    struct MpRecordIdsResult* result) {
  if (result == nullptr) {
    return;
  }
  if (result->ids != nullptr) {
    for (uint32_t i = 0; i < result->ids_count; ++i) {
      free(result->ids[i]);
    }
    delete[] result->ids;
    result->ids = nullptr;
  }
  result->ids_count = 0;
}

absl::Status CppSemanticRetrieverDeleteAll(MpSemanticRetrieverPtr retriever) {
  auto cpp_retriever = GetCppRetriever(retriever);
  return cpp_retriever->DeleteAll();
}

void CppSemanticRetrieverCloseResult(MpRetrievalResult* result) {
  if (result == nullptr) {
    return;
  }
  if (result->records != nullptr) {
    for (uint32_t i = 0; i < result->records_count; ++i) {
      free(result->records[i].id);
      free(result->records[i].text);
      if (result->records[i].metadata != nullptr) {
        for (int j = 0; j < result->records[i].metadata_count; ++j) {
          free(const_cast<char*>(result->records[i].metadata[j].key));
          free(const_cast<char*>(result->records[i].metadata[j].value));
        }
        delete[] result->records[i].metadata;
      }
    }
    delete[] result->records;
    result->records = nullptr;
  }
  result->records_count = 0;
}

absl::Status CppSemanticRetrieverClose(MpSemanticRetrieverPtr retriever) {
  delete retriever;
  return absl::OkStatus();
}

}  // namespace mediapipe::tasks::c::retrieval::semantic_retriever

extern "C" {

MP_EXPORT MpStatus
MpSemanticRetrieverCreate(struct MpSemanticRetrieverOptions* options,
                          MpSemanticRetrieverPtr* retriever, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverCreate(*options, retriever);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSemanticRetrieverInsertDocument(
    MpSemanticRetrieverPtr retriever, const char* id, const char* text,
    struct MpKeyValuePair* metadata, int metadata_count, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverInsertDocument(
          retriever, id ? id : "", text ? text : "", metadata, metadata_count);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSemanticRetrieverInsertImage(
    MpSemanticRetrieverPtr retriever, const char* id,
    const uint8_t* image_bytes, int image_bytes_size, const char* file_path,
    struct MpKeyValuePair* metadata, int metadata_count, char** error_msg) {
  absl::string_view bytes_view;
  if (image_bytes != nullptr && image_bytes_size > 0) {
    bytes_view = absl::string_view(reinterpret_cast<const char*>(image_bytes),
                                   image_bytes_size);
  }
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverInsertImage(retriever, id ? id : "", bytes_view,
                                      file_path ? file_path : "", metadata,
                                      metadata_count);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSemanticRetrieverInsertAudio(
    MpSemanticRetrieverPtr retriever, const char* id, const float* audio_data,
    int audio_data_size, const char* audio_path,
    struct MpKeyValuePair* metadata, int metadata_count, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverInsertAudio(
          retriever, id ? id : "", audio_data, audio_data_size,
          audio_path ? audio_path : "", metadata, metadata_count);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSemanticRetrieverInsertContent(
    MpSemanticRetrieverPtr retriever, const char* id,
    const struct MpTaskPart* parts, int parts_count,
    struct MpKeyValuePair* metadata, int metadata_count, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverInsertContent(retriever, id ? id : "", parts,
                                        parts_count, metadata, metadata_count);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSemanticRetrieverRetrieve(MpSemanticRetrieverPtr retriever,
                                               const struct MpTaskPart* parts,
                                               int parts_count, int limit,
                                               float min_similarity,
                                               MpRetrievalResult* result,
                                               char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverRetrieve(retriever, parts, parts_count, limit,
                                   min_similarity, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSemanticRetrieverRetrieveWithMetadataFilter(
    MpSemanticRetrieverPtr retriever, const struct MpTaskPart* parts,
    int parts_count, int limit, const struct MpKeyValuePair* metadata_filter,
    int metadata_filter_count, float min_similarity, MpRetrievalResult* result,
    char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverRetrieveWithMetadataFilter(
          retriever, parts, parts_count, limit, metadata_filter,
          metadata_filter_count, min_similarity, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSemanticRetrieverDelete(MpSemanticRetrieverPtr retriever,
                                             const char* id, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverDelete(retriever, id ? id : "");
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSemanticRetrieverDeleteWithMetadataFilter(
    MpSemanticRetrieverPtr retriever,
    const struct MpKeyValuePair* metadata_filter, int metadata_filter_count,
    char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverDeleteWithMetadataFilter(retriever, metadata_filter,
                                                   metadata_filter_count);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSemanticRetrieverGetAllRecordIds(
    MpSemanticRetrieverPtr retriever, struct MpRecordIdsResult* result,
    char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverGetAllRecordIds(retriever, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpSemanticRetrieverCloseRecordIdsResult(
    struct MpRecordIdsResult* result) {
  mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverCloseRecordIdsResult(result);
}

MP_EXPORT MpStatus MpSemanticRetrieverDeleteAll(
    MpSemanticRetrieverPtr retriever, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverDeleteAll(retriever);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpSemanticRetrieverCloseResult(MpRetrievalResult* result) {
  mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverCloseResult(result);
}

MP_EXPORT MpStatus MpSemanticRetrieverClose(MpSemanticRetrieverPtr retriever,
                                            char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSemanticRetrieverClose(retriever);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

}  // extern "C"
