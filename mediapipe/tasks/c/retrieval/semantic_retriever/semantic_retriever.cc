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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"           // from @com_google_absl
#include "absl/status/status.h"            // from @com_google_absl
#include "absl/status/status_macros.h"     // from @com_google_absl
#include "absl/status/statusor.h"          // from @com_google_absl
#include "absl/strings/ascii.h"            // from @com_google_absl
#include "absl/strings/str_cat.h"          // from @com_google_absl
#include "absl/strings/string_view.h"      // from @com_google_absl
#include "absl/types/span.h"               // from @com_google_absl
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/retrieval/universal_embedder/universal_embedder_internal.h"
#include "mediapipe/tasks/c/text/text_embedder/text_embedder_internal.h"
#include "mediapipe/tasks/c/vision/image_embedder/image_embedder_internal.h"
#include "mediapipe/tasks/cc/core/embedding_provider.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/memory_store.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/proto/memory.pb.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/semantic_retriever.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/sqlite_memory_store.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/text_chunker.h"
#include "mediapipe/tasks/cc/retrieval/universal_embedder/universal_embedder.h"
#include "mediapipe/tasks/cc/text/text_embedder/text_embedder.h"
#include "mediapipe/tasks/cc/text/utils/genai_utils.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder.h"

struct MpSemanticRetrieverInternal {
  std::unique_ptr<::mediapipe::tasks::retrieval::SemanticRetriever> instance;
};

struct MpSqliteVectorStoreInternal {
  std::unique_ptr<::mediapipe::tasks::retrieval::MemoryStore> store;
  int dimension;
};

namespace mediapipe::tasks::c::retrieval::semantic_retriever {

namespace {

using ::mediapipe::tasks::retrieval::KeyValuePair;
using ::mediapipe::tasks::retrieval::MemoryRecord;
using ::mediapipe::tasks::retrieval::MemoryStore;
using ::mediapipe::tasks::retrieval::Metadata;
using ::mediapipe::tasks::retrieval::Part;
using ::mediapipe::tasks::retrieval::SemanticRetriever;

constexpr int kDefaultChunkSize = 512;
constexpr int kDefaultChunkOverlap = 100;
constexpr int kDefaultEmbeddingDimension = 768;
constexpr int kMaxSqliteVectorDimension =
    (std::numeric_limits<uint16_t>::max() - 4) / sizeof(float);

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

void PopulateRecordIdsResult(const std::vector<std::string>& cpp_ids,
                             struct MpRecordIdsResult* result) {
  result->ids_count = static_cast<uint32_t>(cpp_ids.size());
  if (cpp_ids.empty()) {
    result->ids = nullptr;
    return;
  }
  result->ids = new char*[cpp_ids.size()];
  for (size_t i = 0; i < cpp_ids.size(); ++i) {
    result->ids[i] = strdup(cpp_ids[i].c_str());
  }
}

void PopulateCTaskPartFromProto(const Part& cpp_part, MpTaskPart* out_part) {
  *out_part = MpTaskPart{};
  const std::string& payload = cpp_part.has_text() ? cpp_part.text() : "";
  switch (cpp_part.kind()) {
    case Part::IMAGE:
      out_part->kind = kMpTaskPartImage;
      out_part->image_part.file_path =
          payload.empty() ? nullptr : strdup(payload.c_str());
      break;
    case Part::AUDIO:
      out_part->kind = kMpTaskPartAudio;
      out_part->audio_part.audio_path =
          payload.empty() ? nullptr : strdup(payload.c_str());
      break;
    case Part::TEXT:
    default:
      out_part->kind = kMpTaskPartText;
      out_part->text_part.text = strdup(payload.c_str());
      break;
  }
}

void PopulateVectorStoreRecordsResult(
    const std::vector<MemoryRecord>& cpp_records,
    struct MpVectorStoreRecordsResult* result) {
  result->records_count = static_cast<uint32_t>(cpp_records.size());
  if (cpp_records.empty()) {
    result->records = nullptr;
    return;
  }
  auto* records = new MpVectorStoreRecord[cpp_records.size()];
  for (size_t i = 0; i < cpp_records.size(); ++i) {
    const MemoryRecord& cpp_rec = cpp_records[i];
    MpVectorStoreRecord& c_rec = records[i];
    c_rec = MpVectorStoreRecord{};
    c_rec.id = strdup(cpp_rec.record_id().c_str());

    if (cpp_rec.parts_size() > 0) {
      c_rec.parts_count = cpp_rec.parts_size();
      auto* parts = new MpTaskPart[c_rec.parts_count];
      for (int j = 0; j < c_rec.parts_count; ++j) {
        PopulateCTaskPartFromProto(cpp_rec.parts(j), &parts[j]);
      }
      c_rec.parts = parts;
    } else {
      c_rec.parts_count = 1;
      auto* parts = new MpTaskPart[1];
      parts[0] = MpTaskPart{};
      const std::string& payload = cpp_rec.text();
      if (cpp_rec.content_type() == "IMAGE") {
        parts[0].kind = kMpTaskPartImage;
        parts[0].image_part.file_path =
            payload.empty() ? nullptr : strdup(payload.c_str());
      } else if (cpp_rec.content_type() == "AUDIO") {
        parts[0].kind = kMpTaskPartAudio;
        parts[0].audio_part.audio_path =
            payload.empty() ? nullptr : strdup(payload.c_str());
      } else {
        parts[0].kind = kMpTaskPartText;
        parts[0].text_part.text = strdup(payload.c_str());
      }
      c_rec.parts = parts;
    }

    if (cpp_rec.embeddings_size() > 0) {
      c_rec.embeddings_count = cpp_rec.embeddings_size();
      auto* embeddings = new float[c_rec.embeddings_count];
      std::memcpy(embeddings, cpp_rec.embeddings().data(),
                  sizeof(float) * static_cast<size_t>(c_rec.embeddings_count));
      c_rec.embeddings = embeddings;
    }

    c_rec.has_similarity_score = cpp_rec.has_similarity_score();
    c_rec.similarity_score =
        cpp_rec.has_similarity_score() ? cpp_rec.similarity_score() : 0.0f;

    if (cpp_rec.has_metadata() &&
        !cpp_rec.metadata().key_value_pairs().empty()) {
      c_rec.metadata_count = cpp_rec.metadata().key_value_pairs_size();
      auto* meta = new MpKeyValuePair[c_rec.metadata_count];
      for (int j = 0; j < c_rec.metadata_count; ++j) {
        const KeyValuePair& pair = cpp_rec.metadata().key_value_pairs(j);
        meta[j].key = strdup(pair.key().c_str());
        meta[j].value = strdup(pair.value().c_str());
      }
      c_rec.metadata = meta;
    }

    if (cpp_rec.has_parent_id() && !cpp_rec.parent_id().empty()) {
      c_rec.parent_id = strdup(cpp_rec.parent_id().c_str());
    }

    if (cpp_rec.child_ids_size() > 0) {
      c_rec.child_ids_count = cpp_rec.child_ids_size();
      auto* child_ids = new const char*[c_rec.child_ids_count];
      for (int j = 0; j < c_rec.child_ids_count; ++j) {
        child_ids[j] = strdup(cpp_rec.child_ids(j).c_str());
      }
      c_rec.child_ids = child_ids;
    }
  }
  result->records = records;
}

std::string NormalizeSqliteDatabasePath(const char* database_path) {
  absl::string_view path = absl::NullSafeStringView(database_path);
  return path == ":memory:" ? "" : std::string(path);
}

absl::StatusOr<MemoryRecord> ConvertVectorStoreRecord(
    const MpVectorStoreRecord& c_rec, int expected_dimension) {
  if (c_rec.id == nullptr || c_rec.id[0] == '\0') {
    return absl::InvalidArgumentError("Record id cannot be empty.");
  }
  if (c_rec.parts == nullptr || c_rec.parts_count <= 0) {
    return absl::InvalidArgumentError("Record content cannot be empty.");
  }

  MemoryRecord cpp_record;
  cpp_record.set_record_id(c_rec.id);

  if (c_rec.child_ids != nullptr && c_rec.child_ids_count > 0) {
    for (int j = 0; j < c_rec.child_ids_count; ++j) {
      if (c_rec.child_ids[j] != nullptr && c_rec.child_ids[j][0] != '\0') {
        cpp_record.add_child_ids(c_rec.child_ids[j]);
      }
    }
  }

  const bool is_unembedded_parent = !cpp_record.child_ids().empty() &&
                                    c_rec.embeddings == nullptr &&
                                    c_rec.embeddings_count == 0;
  if (!is_unembedded_parent && (c_rec.embeddings == nullptr ||
                                c_rec.embeddings_count != expected_dimension)) {
    return absl::InvalidArgumentError(
        "Embedding dimension does not match configured dimension.");
  }

  if (c_rec.embeddings != nullptr && c_rec.embeddings_count > 0) {
    cpp_record.mutable_embeddings()->Add(
        c_rec.embeddings, c_rec.embeddings + c_rec.embeddings_count);
  }

  if (c_rec.parts_count == 1) {
    const MpTaskPart& single = c_rec.parts[0];
    switch (single.kind) {
      case kMpTaskPartImage:
        cpp_record.set_content_type("IMAGE");
        cpp_record.set_text(
            absl::NullSafeStringView(single.image_part.file_path));
        break;
      case kMpTaskPartAudio:
        cpp_record.set_content_type("AUDIO");
        cpp_record.set_text(
            absl::NullSafeStringView(single.audio_part.audio_path));
        break;
      case kMpTaskPartText:
      default:
        cpp_record.set_content_type("TEXT");
        cpp_record.set_text(absl::NullSafeStringView(single.text_part.text));
        break;
    }
  }

  for (int j = 0; j < c_rec.parts_count; ++j) {
    const MpTaskPart& c_part = c_rec.parts[j];
    Part* cpp_part = cpp_record.add_parts();
    switch (c_part.kind) {
      case kMpTaskPartImage:
        cpp_part->set_kind(Part::IMAGE);
        cpp_part->set_text(
            absl::NullSafeStringView(c_part.image_part.file_path));
        break;
      case kMpTaskPartAudio:
        cpp_part->set_kind(Part::AUDIO);
        cpp_part->set_text(
            absl::NullSafeStringView(c_part.audio_part.audio_path));
        break;
      case kMpTaskPartText:
      default:
        cpp_part->set_kind(Part::TEXT);
        cpp_part->set_text(absl::NullSafeStringView(c_part.text_part.text));
        break;
    }
  }

  if (c_rec.metadata != nullptr && c_rec.metadata_count > 0) {
    Metadata* cpp_meta = cpp_record.mutable_metadata();
    for (int j = 0; j < c_rec.metadata_count; ++j) {
      if (c_rec.metadata[j].key != nullptr &&
          c_rec.metadata[j].value != nullptr) {
        KeyValuePair* pair = cpp_meta->add_key_value_pairs();
        pair->set_key(c_rec.metadata[j].key);
        pair->set_value(c_rec.metadata[j].value);
      }
    }
  }

  if (c_rec.parent_id != nullptr && c_rec.parent_id[0] != '\0') {
    cpp_record.set_parent_id(c_rec.parent_id);
  }

  return cpp_record;
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

  std::string db_path = NormalizeSqliteDatabasePath(options.database_path);
  ABSL_ASSIGN_OR_RETURN(
      auto memory_store,
      ::mediapipe::tasks::retrieval::SqliteMemoryStore::Create(
          db_path,
          (options.embedding_dimension > 0 ? options.embedding_dimension
                                           : kDefaultEmbeddingDimension)));

  ::mediapipe::tasks::retrieval::SemanticRetrieverComponents components;
  ::mediapipe::tasks::c::core::CppConvertToBaseOptions(
      options.base_options, &components.base_options);
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
  PopulateRecordIdsResult(cpp_ids, result);
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

absl::Status CppSqliteVectorStoreCreate(const char* database_path,
                                        int embedding_dimension,
                                        MpSqliteVectorStorePtr* store) {
  if (store == nullptr) {
    return absl::InvalidArgumentError("Output store pointer cannot be null.");
  }
  *store = nullptr;
  if (embedding_dimension <= 0 ||
      embedding_dimension > kMaxSqliteVectorDimension) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Embedding dimension must be in [1, ", kMaxSqliteVectorDimension,
        "], got ", embedding_dimension, "."));
  }
  std::string normalized_path = NormalizeSqliteDatabasePath(database_path);
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<MemoryStore> memory_store,
      ::mediapipe::tasks::retrieval::SqliteMemoryStore::Create(
          normalized_path, embedding_dimension));
  *store = new MpSqliteVectorStoreInternal{
      .store = std::move(memory_store),
      .dimension = embedding_dimension,
  };
  return absl::OkStatus();
}

absl::Status CppSqliteVectorStoreUpsert(
    MpSqliteVectorStorePtr store, const struct MpVectorStoreRecord* records,
    int records_count) {
  if (store == nullptr || store->store == nullptr) {
    return absl::InvalidArgumentError("Vector store is null or closed.");
  }
  if (records_count == 0) {
    return absl::OkStatus();
  }
  if (records == nullptr || records_count < 0) {
    return absl::InvalidArgumentError(
        "Records pointer must be non-null when records_count is non-zero.");
  }

  std::vector<MemoryRecord> cpp_records;
  cpp_records.reserve(static_cast<size_t>(records_count));
  for (int i = 0; i < records_count; ++i) {
    ABSL_ASSIGN_OR_RETURN(
        MemoryRecord cpp_record,
        ConvertVectorStoreRecord(records[i], store->dimension));
    cpp_records.push_back(std::move(cpp_record));
  }

  absl::flat_hash_map<std::string, size_t> last_index_by_id;
  last_index_by_id.reserve(cpp_records.size());
  for (size_t i = 0; i < cpp_records.size(); ++i) {
    last_index_by_id[cpp_records[i].record_id()] = i;
  }

  std::vector<MemoryRecord> deduped_records;
  deduped_records.reserve(last_index_by_id.size());
  std::vector<std::string> distinct_ids;
  distinct_ids.reserve(last_index_by_id.size());
  for (size_t i = 0; i < cpp_records.size(); ++i) {
    if (last_index_by_id[cpp_records[i].record_id()] == i) {
      distinct_ids.push_back(cpp_records[i].record_id());
      deduped_records.push_back(std::move(cpp_records[i]));
    }
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<MemoryRecord> existing_records,
                        store->store->GetRecords(distinct_ids));
  std::vector<std::string> ids_to_delete = distinct_ids;
  for (const MemoryRecord& existing : existing_records) {
    for (const std::string& child_id : existing.child_ids()) {
      if (!child_id.empty() &&
          last_index_by_id.try_emplace(child_id, cpp_records.size()).second) {
        ids_to_delete.push_back(child_id);
      }
    }
  }
  for (const std::string& id_to_delete : ids_to_delete) {
    ABSL_RETURN_IF_ERROR(store->store->DeleteById(id_to_delete));
  }

  return store->store->InsertBatch(absl::MakeConstSpan(deduped_records));
}

absl::Status CppSqliteVectorStoreSearch(
    MpSqliteVectorStorePtr store, const float* query_embedding,
    int query_embedding_size, int limit, float min_similarity_score,
    const struct MpKeyValuePair* metadata_filter, int metadata_filter_count,
    struct MpVectorStoreRecordsResult* result) {
  if (result == nullptr) {
    return absl::InvalidArgumentError("Output result pointer cannot be null.");
  }
  *result = MpVectorStoreRecordsResult{};
  if (store == nullptr || store->store == nullptr) {
    return absl::InvalidArgumentError("Vector store is null or closed.");
  }
  if (query_embedding == nullptr || query_embedding_size <= 0 || limit <= 0) {
    return absl::InvalidArgumentError(
        "query_embedding must be non-empty and limit must be positive.");
  }
  if (query_embedding_size != store->dimension) {
    return absl::InvalidArgumentError(
        "Embedding dimension does not match configured dimension.");
  }

  std::vector<float> cpp_query(query_embedding,
                               query_embedding + query_embedding_size);
  absl::flat_hash_map<std::string, std::string> cpp_filter =
      ConvertMetadataMap(metadata_filter, metadata_filter_count);
  ABSL_ASSIGN_OR_RETURN(
      std::vector<MemoryRecord> cpp_records,
      store->store->GetNearestRecords(std::move(cpp_query), limit,
                                      min_similarity_score, cpp_filter));
  PopulateVectorStoreRecordsResult(cpp_records, result);
  return absl::OkStatus();
}

absl::Status CppSqliteVectorStoreGetRecords(
    MpSqliteVectorStorePtr store, const char* const* ids, int ids_count,
    struct MpVectorStoreRecordsResult* result) {
  if (result == nullptr) {
    return absl::InvalidArgumentError("Output result pointer cannot be null.");
  }
  *result = MpVectorStoreRecordsResult{};
  if (store == nullptr || store->store == nullptr) {
    return absl::InvalidArgumentError("Vector store is null or closed.");
  }
  if (ids_count == 0) {
    return absl::OkStatus();
  }
  if (ids == nullptr || ids_count < 0) {
    return absl::InvalidArgumentError(
        "ids must be non-null when ids_count is non-zero.");
  }

  std::vector<std::string> cpp_ids;
  cpp_ids.reserve(static_cast<size_t>(ids_count));
  for (int i = 0; i < ids_count; ++i) {
    cpp_ids.emplace_back(ids[i] ? ids[i] : "");
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<MemoryRecord> cpp_records,
                        store->store->GetRecords(cpp_ids));
  PopulateVectorStoreRecordsResult(cpp_records, result);
  return absl::OkStatus();
}

absl::Status CppSqliteVectorStoreGetAllRecordIds(
    MpSqliteVectorStorePtr store, struct MpRecordIdsResult* result) {
  if (result == nullptr) {
    return absl::InvalidArgumentError("Output result pointer cannot be null.");
  }
  *result = MpRecordIdsResult{};
  if (store == nullptr || store->store == nullptr) {
    return absl::InvalidArgumentError("Vector store is null or closed.");
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<std::string> cpp_ids,
                        store->store->GetAllRecordIds());
  PopulateRecordIdsResult(cpp_ids, result);
  return absl::OkStatus();
}

absl::Status CppSqliteVectorStoreDeleteByIds(MpSqliteVectorStorePtr store,
                                             const char* const* ids,
                                             int ids_count) {
  if (store == nullptr || store->store == nullptr) {
    return absl::InvalidArgumentError("Vector store is null or closed.");
  }
  if (ids_count == 0) {
    return absl::OkStatus();
  }
  if (ids == nullptr || ids_count < 0) {
    return absl::InvalidArgumentError(
        "ids must be non-null when ids_count is non-zero.");
  }
  for (int i = 0; i < ids_count; ++i) {
    ABSL_RETURN_IF_ERROR(store->store->DeleteById(ids[i] ? ids[i] : ""));
  }
  return absl::OkStatus();
}

absl::Status CppSqliteVectorStoreDeleteByMetadata(
    MpSqliteVectorStorePtr store, const struct MpKeyValuePair* metadata_filter,
    int metadata_filter_count) {
  if (store == nullptr || store->store == nullptr) {
    return absl::InvalidArgumentError("Vector store is null or closed.");
  }
  absl::flat_hash_map<std::string, std::string> cpp_filter =
      ConvertMetadataMap(metadata_filter, metadata_filter_count);
  if (cpp_filter.empty()) {
    return absl::InvalidArgumentError("metadata_filter cannot be empty.");
  }
  return store->store->DeleteByMetadata(cpp_filter);
}

absl::Status CppSqliteVectorStoreDeleteAll(MpSqliteVectorStorePtr store) {
  if (store == nullptr || store->store == nullptr) {
    return absl::InvalidArgumentError("Vector store is null or closed.");
  }
  return store->store->DeleteAll();
}

void CppSqliteVectorStoreCloseRecordsResult(
    struct MpVectorStoreRecordsResult* result) {
  if (result == nullptr) {
    return;
  }
  if (result->records != nullptr) {
    for (uint32_t i = 0; i < result->records_count; ++i) {
      MpVectorStoreRecord& rec = result->records[i];
      free(const_cast<char*>(rec.id));
      if (rec.parts != nullptr) {
        for (int j = 0; j < rec.parts_count; ++j) {
          free(const_cast<char*>(rec.parts[j].text_part.text));
          free(const_cast<char*>(rec.parts[j].image_part.file_path));
          free(const_cast<char*>(rec.parts[j].audio_part.audio_path));
        }
        delete[] rec.parts;
      }
      delete[] rec.embeddings;
      if (rec.metadata != nullptr) {
        for (int j = 0; j < rec.metadata_count; ++j) {
          free(const_cast<char*>(rec.metadata[j].key));
          free(const_cast<char*>(rec.metadata[j].value));
        }
        delete[] rec.metadata;
      }
      free(const_cast<char*>(rec.parent_id));
      if (rec.child_ids != nullptr) {
        for (int j = 0; j < rec.child_ids_count; ++j) {
          free(const_cast<char*>(rec.child_ids[j]));
        }
        delete[] rec.child_ids;
      }
    }
    delete[] result->records;
    result->records = nullptr;
  }
  result->records_count = 0;
}

absl::Status CppSqliteVectorStoreClose(MpSqliteVectorStorePtr store) {
  delete store;
  return absl::OkStatus();
}

absl::Status CppTextChunkerChunk(absl::string_view text, int chunk_size,
                                 int chunk_overlap, int word_threshold,
                                 enum MpChunkingMode mode,
                                 struct MpTextChunksResult* result) {
  if (result == nullptr) {
    return absl::InvalidArgumentError("Output result pointer cannot be null.");
  }
  *result = MpTextChunksResult{};

  if (chunk_size <= 0) {
    return absl::InvalidArgumentError("Chunk size must be positive.");
  }
  if (mode == kMpChunkingModeCharacter) {
    if (chunk_overlap < 0 || chunk_overlap >= chunk_size) {
      return absl::InvalidArgumentError(
          "Chunk overlap must be non-negative and less than chunk size.");
    }
  } else if (mode == kMpChunkingModeWord) {
    if (word_threshold <= 0) {
      return absl::InvalidArgumentError("Word threshold must be positive.");
    }
    if (chunk_size > word_threshold) {
      return absl::InvalidArgumentError(
          "Chunk size (word chunk size) must be less than or equal to word "
          "threshold.");
    }
  } else {
    return absl::InvalidArgumentError("Unsupported chunking mode.");
  }

  if (absl::StripAsciiWhitespace(text).empty()) {
    return absl::OkStatus();
  }

  std::vector<std::string> cpp_chunks;
  if (mode == kMpChunkingModeCharacter) {
    cpp_chunks = ::mediapipe::tasks::text::utils::ChunkTextByCountingCharacters(
        text, chunk_size, chunk_overlap);
  } else {
    std::vector<::mediapipe::tasks::text::utils::TextChunk> raw_chunks =
        ::mediapipe::tasks::text::utils::ChunkText(text, word_threshold,
                                                   chunk_size);
    cpp_chunks.reserve(raw_chunks.size());
    for (const ::mediapipe::tasks::text::utils::TextChunk& chunk : raw_chunks) {
      cpp_chunks.emplace_back(chunk.text);
    }
  }

  result->chunks_count = static_cast<uint32_t>(cpp_chunks.size());
  if (!cpp_chunks.empty()) {
    result->chunks = new char*[cpp_chunks.size()];
    for (size_t i = 0; i < cpp_chunks.size(); ++i) {
      result->chunks[i] = strdup(cpp_chunks[i].c_str());
    }
  }
  return absl::OkStatus();
}

void CppTextChunkerCloseResult(struct MpTextChunksResult* result) {
  if (result == nullptr) {
    return;
  }
  if (result->chunks != nullptr) {
    for (uint32_t i = 0; i < result->chunks_count; ++i) {
      free(result->chunks[i]);
    }
    delete[] result->chunks;
    result->chunks = nullptr;
  }
  result->chunks_count = 0;
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

MP_EXPORT MpStatus MpSqliteVectorStoreCreate(const char* database_path,
                                             int embedding_dimension,
                                             MpSqliteVectorStorePtr* store,
                                             char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSqliteVectorStoreCreate(database_path, embedding_dimension, store);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSqliteVectorStoreUpsert(
    MpSqliteVectorStorePtr store, const struct MpVectorStoreRecord* records,
    int records_count, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSqliteVectorStoreUpsert(store, records, records_count);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSqliteVectorStoreSearch(
    MpSqliteVectorStorePtr store, const float* query_embedding,
    int query_embedding_size, int limit, float min_similarity_score,
    const struct MpKeyValuePair* metadata_filter, int metadata_filter_count,
    struct MpVectorStoreRecordsResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSqliteVectorStoreSearch(store, query_embedding, query_embedding_size,
                                 limit, min_similarity_score, metadata_filter,
                                 metadata_filter_count, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSqliteVectorStoreGetRecords(
    MpSqliteVectorStorePtr store, const char* const* ids, int ids_count,
    struct MpVectorStoreRecordsResult* result, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSqliteVectorStoreGetRecords(store, ids, ids_count, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSqliteVectorStoreGetAllRecordIds(
    MpSqliteVectorStorePtr store, struct MpRecordIdsResult* result,
    char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSqliteVectorStoreGetAllRecordIds(store, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSqliteVectorStoreDeleteByIds(MpSqliteVectorStorePtr store,
                                                  const char* const* ids,
                                                  int ids_count,
                                                  char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSqliteVectorStoreDeleteByIds(store, ids, ids_count);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSqliteVectorStoreDeleteByMetadata(
    MpSqliteVectorStorePtr store, const struct MpKeyValuePair* metadata_filter,
    int metadata_filter_count, char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSqliteVectorStoreDeleteByMetadata(store, metadata_filter,
                                           metadata_filter_count);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpSqliteVectorStoreDeleteAll(MpSqliteVectorStorePtr store,
                                                char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSqliteVectorStoreDeleteAll(store);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpSqliteVectorStoreCloseRecordsResult(
    struct MpVectorStoreRecordsResult* result) {
  mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSqliteVectorStoreCloseRecordsResult(result);
}

MP_EXPORT MpStatus MpSqliteVectorStoreClose(MpSqliteVectorStorePtr store,
                                            char** error_msg) {
  absl::Status status = mediapipe::tasks::c::retrieval::semantic_retriever::
      CppSqliteVectorStoreClose(store);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpTextChunkerChunk(const char* text, int chunk_size,
                                      int chunk_overlap, int word_threshold,
                                      enum MpChunkingMode mode,
                                      struct MpTextChunksResult* result,
                                      char** error_msg) {
  absl::Status status =
      mediapipe::tasks::c::retrieval::semantic_retriever::CppTextChunkerChunk(
          absl::NullSafeStringView(text), chunk_size, chunk_overlap,
          word_threshold, mode, result);
  return mediapipe::tasks::c::core::HandleStatus(status, error_msg);
}

MP_EXPORT void MpTextChunkerCloseResult(struct MpTextChunksResult* result) {
  mediapipe::tasks::c::retrieval::semantic_retriever::CppTextChunkerCloseResult(
      result);
}

}  // extern "C"
