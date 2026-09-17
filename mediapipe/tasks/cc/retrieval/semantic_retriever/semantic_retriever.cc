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

#include "mediapipe/tasks/cc/retrieval/semantic_retriever/semantic_retriever.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/core/embedding_provider.h"
#include "mediapipe/tasks/cc/core/logging/factory/logging_factory.h"
#include "mediapipe/tasks/cc/core/running_mode.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/audio_decoder.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/image_decoder.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/memory_store.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/text_chunker.h"

namespace mediapipe::tasks::retrieval {
namespace {

constexpr int kDefaultChunkSize = 512;
constexpr int kDefaultChunkOverlap = 100;

absl::StatusOr<std::vector<core::TaskPart>> FillTaskParts(
    const std::vector<core::TaskPart>& parts) {
  std::vector<core::TaskPart> resolved_parts;
  resolved_parts.reserve(parts.size());

  for (const auto& part : parts) {
    if (std::holds_alternative<core::TextPart>(part)) {
      resolved_parts.push_back(part);
    } else if (std::holds_alternative<core::ImagePart>(part)) {
      const auto& image_part = std::get<core::ImagePart>(part);
      if (image_part.image_bytes.empty()) {
        ABSL_ASSIGN_OR_RETURN(auto image_bytes, ImageDecoder::DecodeImageBytes(
                                                    image_part.file_path));
        resolved_parts.push_back(
            core::ImagePart(image_part.file_path, std::move(image_bytes)));
      } else {
        resolved_parts.push_back(part);
      }
    } else if (std::holds_alternative<core::AudioPart>(part)) {
      const auto& audio_part = std::get<core::AudioPart>(part);
      if (audio_part.audio_data.empty()) {
        ABSL_ASSIGN_OR_RETURN(auto audio_data, AudioDecoder::DecodeAudioData(
                                                   audio_part.file_path));
        resolved_parts.push_back(
            core::AudioPart(audio_part.file_path, std::move(audio_data)));
      } else {
        resolved_parts.push_back(part);
      }
    }
  }
  return resolved_parts;
}

}  // namespace

absl::StatusOr<std::unique_ptr<SemanticRetriever>> SemanticRetriever::Create(
    SemanticRetrieverComponents components) {
  if (components.memory_store == nullptr) {
    return absl::InvalidArgumentError("MemoryStore must not be null.");
  }
  if (components.embedding_provider == nullptr) {
    return absl::InvalidArgumentError("EmbeddingProvider must not be null.");
  }

  auto tasks_logger = core::logging::CreateTasksLogger(
      {.task_name = "SemanticRetriever",
       .task_running_mode = core::RunningMode::kUnspecified});
  tasks_logger->LogSessionStart();

  std::unique_ptr<TextChunker> text_chunker =
      std::move(components.text_chunker);
  if (text_chunker == nullptr) {
    text_chunker = std::make_unique<DefaultTextChunker>(
        kDefaultChunkSize, kDefaultChunkOverlap, ChunkingMode::kCharacter);
  }

  return absl::WrapUnique(
      new SemanticRetriever(std::move(components.memory_store),
                            std::move(components.embedding_provider),
                            std::move(text_chunker), std::move(tasks_logger)));
}

SemanticRetriever::~SemanticRetriever() { tasks_logger_->LogSessionEnd(); }

void AddMetadataToRecord(
    MemoryRecord* record,
    const absl::flat_hash_map<std::string, std::string>& metadata) {
  if (metadata.empty()) {
    return;
  }
  for (const auto& [key, value] : metadata) {
    auto* pair = record->mutable_metadata()->add_key_value_pairs();
    pair->set_key(key);
    pair->set_value(value);
  }
}

absl::Status SemanticRetriever::InsertDocument(
    absl::string_view id, absl::string_view text,
    const absl::flat_hash_map<std::string, std::string>& metadata) {
  // Delete any existing document or chunks with this ID to prevent orphaned
  // chunks.
  ABSL_RETURN_IF_ERROR(memory_store_->DeleteById(id));

  ABSL_ASSIGN_OR_RETURN(std::vector<std::string> chunks,
                        text_chunker_->Chunk(text));

  bool has_chunks = chunks.size() > 1;

  if (!has_chunks) {
    MemoryRecord record;
    std::vector<core::TaskPart> parts;
    parts.push_back(core::TextPart{std::string(text)});
    ABSL_ASSIGN_OR_RETURN(auto embedding_opt,
                          embedding_provider_->EmbedContent(parts));
    if (!embedding_opt.has_value() || embedding_opt->empty()) {
      return absl::InternalError("Failed to extract embedding.");
    }

    record.set_text(text);
    for (float val : *embedding_opt) {
      record.add_embeddings(val);
    }

    record.set_record_id(id);
    record.set_content_type("TEXT");

    AddMetadataToRecord(&record, metadata);

    ABSL_RETURN_IF_ERROR(memory_store_->Insert(record));
  } else {
    MemoryRecord parent_record;
    parent_record.set_text(text);
    parent_record.set_record_id(id);
    parent_record.set_content_type("TEXT");
    AddMetadataToRecord(&parent_record, metadata);

    for (size_t i = 0; i < chunks.size(); ++i) {
      MemoryRecord record;
      std::vector<core::TaskPart> parts;
      parts.push_back(core::TextPart{chunks[i]});
      ABSL_ASSIGN_OR_RETURN(auto embedding_opt,
                            embedding_provider_->EmbedContent(parts));
      if (!embedding_opt.has_value() || embedding_opt->empty()) {
        return absl::InternalError("Failed to extract embedding.");
      }

      record.set_text(chunks[i]);
      for (float val : *embedding_opt) {
        record.add_embeddings(val);
      }

      std::string chunk_id = absl::StrCat(id, "_chunk_", i);
      record.set_record_id(chunk_id);
      record.set_parent_id(id);
      record.set_content_type("TEXT");

      AddMetadataToRecord(&record, metadata);

      ABSL_RETURN_IF_ERROR(memory_store_->Insert(record));
      parent_record.add_child_ids(std::move(chunk_id));
    }
    ABSL_RETURN_IF_ERROR(memory_store_->Insert(parent_record));
  }

  return absl::OkStatus();
}

absl::Status SemanticRetriever::InsertContent(
    absl::string_view id, const std::vector<core::TaskPart>& parts,
    const absl::flat_hash_map<std::string, std::string>& metadata) {
  // Delete any existing document or chunks with this ID to prevent orphaned
  // chunks.
  ABSL_RETURN_IF_ERROR(memory_store_->DeleteById(id));

  // Resolve any empty media parts by loading/decoding them from their file
  // paths
  ABSL_ASSIGN_OR_RETURN(std::vector<core::TaskPart> resolved_parts,
                        FillTaskParts(parts));

  // Generate a single combined embedding for the entire resolved multimodal
  // content
  ABSL_ASSIGN_OR_RETURN(auto embedding_opt,
                        embedding_provider_->EmbedContent(resolved_parts));
  if (!embedding_opt.has_value() || embedding_opt->empty()) {
    return absl::InternalError("Failed to extract multimodal embedding.");
  }

  MemoryRecord record;
  for (float val : *embedding_opt) {
    record.add_embeddings(val);
  }

  record.set_record_id(id);

  std::string content_type = "TEXT";
  if (!resolved_parts.empty()) {
    if (std::holds_alternative<core::ImagePart>(resolved_parts[0])) {
      content_type = "IMAGE";
    } else if (std::holds_alternative<core::AudioPart>(resolved_parts[0])) {
      content_type = "AUDIO";
    }
  }
  record.set_content_type(content_type);

  AddMetadataToRecord(&record, metadata);

  // Populate parts and primary text field
  std::string combined_text;
  for (const auto& part : resolved_parts) {
    std::string part_content;
    if (std::holds_alternative<core::TextPart>(part)) {
      part_content = std::get<core::TextPart>(part).text;
    } else if (std::holds_alternative<core::ImagePart>(part)) {
      part_content = std::get<core::ImagePart>(part).file_path;
    } else if (std::holds_alternative<core::AudioPart>(part)) {
      part_content = std::get<core::AudioPart>(part).file_path;
    }

    if (!part_content.empty()) {
      if (!combined_text.empty()) {
        absl::StrAppend(&combined_text, " ");
      }
      absl::StrAppend(&combined_text, part_content);
    }
  }

  for (const auto& part : resolved_parts) {
    if (std::holds_alternative<core::TextPart>(part)) {
      const auto& text_part = std::get<core::TextPart>(part);
      auto* record_part = record.add_parts();
      record_part->set_kind(Part::TEXT);
      record_part->set_text(text_part.text);
    } else if (std::holds_alternative<core::ImagePart>(part)) {
      const auto& image_part = std::get<core::ImagePart>(part);
      auto* record_part = record.add_parts();
      record_part->set_kind(Part::IMAGE);
      record_part->set_text(image_part.file_path);
    } else if (std::holds_alternative<core::AudioPart>(part)) {
      const auto& audio_part = std::get<core::AudioPart>(part);
      auto* record_part = record.add_parts();
      record_part->set_kind(Part::AUDIO);
      record_part->set_text(audio_part.file_path);
    }
  }

  record.set_text(combined_text);
  ABSL_RETURN_IF_ERROR(memory_store_->Insert(record));
  return absl::OkStatus();
}

absl::Status SemanticRetriever::InsertImage(
    absl::string_view id, absl::string_view file_path,
    const absl::flat_hash_map<std::string, std::string>& metadata) {
  std::vector<core::TaskPart> parts;
  parts.push_back(core::ImagePart{std::string(file_path), ""});
  return InsertContent(id, parts, metadata);
}

absl::Status SemanticRetriever::InsertImage(
    absl::string_view id, absl::string_view image_bytes,
    absl::string_view file_path,
    const absl::flat_hash_map<std::string, std::string>& metadata) {
  std::vector<core::TaskPart> parts;
  parts.push_back(
      core::ImagePart{std::string(file_path), std::string(image_bytes)});
  return InsertContent(id, parts, metadata);
}

absl::Status SemanticRetriever::InsertAudio(
    absl::string_view id, absl::string_view file_path,
    const absl::flat_hash_map<std::string, std::string>& metadata) {
  std::vector<core::TaskPart> parts;
  parts.push_back(core::AudioPart{std::string(file_path), {}});
  return InsertContent(id, parts, metadata);
}

absl::Status SemanticRetriever::InsertAudio(
    absl::string_view id, const std::vector<float>& audio_data,
    absl::string_view file_path,
    const absl::flat_hash_map<std::string, std::string>& metadata) {
  std::vector<core::TaskPart> parts;
  parts.push_back(core::AudioPart{std::string(file_path), audio_data});
  return InsertContent(id, parts, metadata);
}

absl::StatusOr<std::vector<MemoryRecord>> SemanticRetriever::Retrieve(
    const std::vector<core::TaskPart>& query_content, int limit,
    const absl::flat_hash_map<std::string, std::string>& metadata_filter,
    float min_similarity) {
  mediapipe::Timestamp task_logger_ts(tasks_logger_timestamp_++);
  tasks_logger_->RecordCpuInputArrival(task_logger_ts);
  absl::Cleanup log_invocation_end = [this, task_logger_ts] {
    tasks_logger_->RecordInvocationEnd(task_logger_ts);
  };

  ABSL_ASSIGN_OR_RETURN(auto embedding_opt,
                        embedding_provider_->EmbedContent(query_content));
  if (!embedding_opt.has_value() || embedding_opt->empty()) {
    return absl::InternalError("Failed to extract query embedding.");
  }

  // Oversample candidates to avoid under-fetching during parent deduplication
  int query_limit = limit * 3;
  ABSL_ASSIGN_OR_RETURN(
      auto search_results,
      memory_store_->GetNearestRecords(*embedding_opt, query_limit,
                                       min_similarity, metadata_filter));

  // Collect unique parent IDs into a single set
  absl::flat_hash_set<std::string> parent_ids_to_fetch;
  for (const auto& record : search_results) {
    if (!record.parent_id().empty()) {
      parent_ids_to_fetch.insert(record.parent_id());
    }
  }

  // Fetch all parent records
  absl::flat_hash_map<std::string, MemoryRecord> parent_records_map;
  if (!parent_ids_to_fetch.empty()) {
    std::vector<std::string> fetch_list(parent_ids_to_fetch.begin(),
                                        parent_ids_to_fetch.end());
    ABSL_ASSIGN_OR_RETURN(auto parent_records,
                          memory_store_->GetRecords(fetch_list));

    for (auto& parent_rec : parent_records) {
      std::string rec_id = parent_rec.record_id();
      parent_records_map[std::move(rec_id)] = std::move(parent_rec);
    }
  }

  // Build deduplicated final results preserving rank similarity order
  std::vector<MemoryRecord> final_results;
  absl::flat_hash_set<std::string> emitted_ids;

  for (const auto& record : search_results) {
    std::string record_id = record.record_id();

    if (!record.parent_id().empty()) {
      const std::string& parent_id = record.parent_id();
      if (emitted_ids.insert(parent_id).second) {
        auto it = parent_records_map.find(parent_id);
        if (it != parent_records_map.end()) {
          MemoryRecord parent_record = std::move(it->second);
          parent_record.set_similarity_score(record.similarity_score());
          parent_record.clear_embeddings();
          for (float val : record.embeddings()) {
            parent_record.add_embeddings(val);
          }
          final_results.push_back(std::move(parent_record));
        }
      }
    } else {
      if (emitted_ids.insert(record_id).second) {
        final_results.push_back(record);
      }
    }

    if (final_results.size() >= limit) {
      break;
    }
  }

  return final_results;
}

absl::StatusOr<std::vector<MemoryRecord>> SemanticRetriever::Retrieve(
    const std::vector<core::TaskPart>& query_content, int limit,
    float min_similarity) {
  return Retrieve(query_content, limit, /*metadata_filter=*/{}, min_similarity);
}

absl::StatusOr<std::vector<MemoryRecord>> SemanticRetriever::Retrieve(
    absl::string_view query, int limit,
    const absl::flat_hash_map<std::string, std::string>& metadata_filter,
    float min_similarity) {
  std::vector<core::TaskPart> parts;
  parts.push_back(core::TextPart{std::string(query)});
  return Retrieve(parts, limit, metadata_filter, min_similarity);
}

absl::StatusOr<std::vector<MemoryRecord>> SemanticRetriever::Retrieve(
    absl::string_view query, int limit, float min_similarity) {
  return Retrieve(query, limit, /*metadata_filter=*/{}, min_similarity);
}

absl::Status SemanticRetriever::Delete(absl::string_view id) {
  return memory_store_->DeleteById(id);
}

absl::Status SemanticRetriever::Delete(
    const absl::flat_hash_map<std::string, std::string>& metadata_filter) {
  return memory_store_->DeleteByMetadata(metadata_filter);
}

absl::StatusOr<std::vector<std::string>> SemanticRetriever::GetAllRecordIds() {
  return memory_store_->GetAllRecordIds();
}

absl::Status SemanticRetriever::DeleteAll() {
  return memory_store_->DeleteAll();
}

}  // namespace mediapipe::tasks::retrieval
