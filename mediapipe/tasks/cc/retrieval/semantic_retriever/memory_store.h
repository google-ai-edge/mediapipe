// Copyright 2025 The Google AI Edge Authors.
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

#ifndef THIRD_PARTY_ODML_LLM_EXTENSIONS_RAG_PIPELINE_CORE_MEMORY_MEMORY_STORE_H_
#define THIRD_PARTY_ODML_LLM_EXTENSIONS_RAG_PIPELINE_CORE_MEMORY_MEMORY_STORE_H_

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/proto/memory.pb.h"

namespace mediapipe {
namespace tasks {
namespace retrieval {

// An interface for the data store holding memory records that containing text
// and embeddings.
class MemoryStore {
 public:
  virtual absl::Status Insert(MemoryRecord record) = 0;

  // Inserts a batch of records into the memory store for better concurrency
  // processing.
  virtual absl::Status InsertBatch(absl::Span<const MemoryRecord> records) = 0;

  virtual absl::StatusOr<std::vector<MemoryRecord>> GetNearestRecords(
      std::vector<float> queryEmbeddings, int topK,
      float minSimilarityScore) = 0;

  virtual absl::StatusOr<std::vector<MemoryRecord>> GetNearestRecords(
      std::vector<float> queryEmbeddings, int topK, float minSimilarityScore,
      const absl::flat_hash_map<std::string, std::string>& metadata_filter) {
    if (metadata_filter.empty()) {
      return GetNearestRecords(std::move(queryEmbeddings), topK,
                               minSimilarityScore);
    }
    ABSL_ASSIGN_OR_RETURN(auto records,
                          GetNearestRecords(std::move(queryEmbeddings), topK,
                                            minSimilarityScore));
    std::vector<MemoryRecord> filtered_records;
    for (const auto& record : records) {
      if (!record.has_metadata()) {
        continue;
      }
      bool match = std::all_of(
          metadata_filter.begin(), metadata_filter.end(),
          [&](const auto& filter_pair) {
            return std::any_of(record.metadata().key_value_pairs().begin(),
                               record.metadata().key_value_pairs().end(),
                               [&](const auto& pair) {
                                 return pair.key() == filter_pair.first &&
                                        pair.value() == filter_pair.second;
                               });
          });
      if (match) {
        filtered_records.push_back(record);
      }
    }
    return filtered_records;
  }

  // Deletes records matching the given metadata key-value filter.
  virtual absl::Status DeleteByMetadata(
      const absl::flat_hash_map<std::string, std::string>& metadata_filter) {
    if (metadata_filter.empty()) {
      return absl::InvalidArgumentError("metadata_filter cannot be empty.");
    }
    ABSL_ASSIGN_OR_RETURN(auto record_pairs, GetAllRecords());
    for (const auto& [id, record] : record_pairs) {
      if (!record.has_metadata()) {
        continue;
      }
      bool match = std::all_of(
          metadata_filter.begin(), metadata_filter.end(),
          [&](const auto& filter_pair) {
            return std::any_of(record.metadata().key_value_pairs().begin(),
                               record.metadata().key_value_pairs().end(),
                               [&](const auto& pair) {
                                 return pair.key() == filter_pair.first &&
                                        pair.value() == filter_pair.second;
                               });
          });
      if (match) {
        ABSL_RETURN_IF_ERROR(DeleteById(id));
      }
    }
    return absl::OkStatus();
  }

  // Deletes records that match the given text.
  virtual absl::Status Delete(absl::string_view text) = 0;

  // Retrieves records that match the given IDs.
  virtual absl::StatusOr<std::vector<MemoryRecord>> GetRecords(
      const std::vector<std::string>& ids) = 0;

  // Deletes a record with the given ID.
  virtual absl::Status DeleteById(absl::string_view id) = 0;

  // Retrieves all unique root records and their IDs in the store.
  virtual absl::StatusOr<std::vector<std::pair<std::string, MemoryRecord>>>
  GetAllRecords() = 0;

  // Retrieves all unique record_ids in the store.
  virtual absl::StatusOr<std::vector<std::string>> GetAllRecordIds() = 0;

  // Deletes all records in the store.
  virtual absl::Status DeleteAll() = 0;

  virtual ~MemoryStore() = default;
};

}  // namespace retrieval
}  // namespace tasks
}  // namespace mediapipe

#endif  // THIRD_PARTY_ODML_LLM_EXTENSIONS_RAG_PIPELINE_CORE_MEMORY_MEMORY_STORE_H_
