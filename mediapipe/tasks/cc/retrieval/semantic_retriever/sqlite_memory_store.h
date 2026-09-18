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

#ifndef MEDIAPIPE_TASKS_CC_RETRIEVAL_SQLITE_MEMORY_STORE_H_
#define MEDIAPIPE_TASKS_CC_RETRIEVAL_SQLITE_MEMORY_STORE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/memory_store.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/sqlite_vector_store.h"

namespace mediapipe {
namespace tasks {
namespace retrieval {

// A memory store implementation using SQLite vector store.
class SqliteMemoryStore : public MemoryStore {
 public:
  struct Config {
    // Name of the column that stores the source text.
    std::string text_column_name;
    // Optional name of the column that stores image bytes.
    std::string image_column_name;
    // Optional name of the column that stores audio bytes.
    std::string audio_column_name;
    // Name of the column that stores the embedding vector.
    std::string embedding_column_name;
    // Dimension of the embedding vector.
    int embedding_dimension;
    TableConfig table_config;
  };

  static Config GetDefaultConfig(int embedding_dimension);

  // Creates a file-backed SQLite store.
  static absl::StatusOr<std::unique_ptr<MemoryStore>> Create(
      absl::string_view database_path, int embedding_dimension);

  // Creates an in-memory SQLite store.
  static absl::StatusOr<std::unique_ptr<MemoryStore>> CreateInMemory(
      int embedding_dimension);

  explicit SqliteMemoryStore(std::unique_ptr<SqliteVectorStore> sqlite_store)
      : sqlite_store_(std::move(sqlite_store)) {}

  SqliteVectorStore* sqlite_store() const { return sqlite_store_.get(); }
  // Initializes the memory store with the given table name and embedding
  // metadata. The table with the given name is used to store the texts,
  // embeddings and other information. Will create a new table if it doesn't
  // already exist.
  absl::Status Initialize(const Config& config);
  absl::Status Insert(MemoryRecord record) override;
  absl::StatusOr<std::vector<MemoryRecord>> GetNearestRecords(
      std::vector<float> query_embeddings, int top_k,
      float min_similarity_score) override;
  absl::StatusOr<std::vector<MemoryRecord>> GetNearestRecords(
      std::vector<float> query_embeddings, int top_k,
      float min_similarity_score,
      const absl::flat_hash_map<std::string, std::string>& metadata_filter)
      override;
  absl::StatusOr<std::vector<MemoryRecord>> GetRecords(
      const std::vector<std::string>& ids) override;
  absl::Status Delete(absl::string_view text) override;
  absl::Status DeleteById(absl::string_view id) override;
  absl::Status DeleteByMetadata(
      const absl::flat_hash_map<std::string, std::string>& metadata_filter)
      override;
  absl::StatusOr<std::vector<std::pair<std::string, MemoryRecord>>>
  GetAllRecords() override;
  absl::StatusOr<std::vector<std::string>> GetAllRecordIds() override;
  absl::Status DeleteAll() override;

  absl::Status InsertBatch(absl::Span<const MemoryRecord> records) override;

  // Executes a SQL query that does not return results.
  absl::Status Execute(std::string sql_query);

  SqliteMemoryStore() = delete;
  SqliteMemoryStore(const SqliteMemoryStore&) = delete;
  SqliteMemoryStore& operator=(const SqliteMemoryStore&) = delete;

 private:
  MemoryRecord ToMemoryRecord(
      const std::vector<SqliteVectorStore::ColumnValue>& column_values) const;

  std::unique_ptr<SqliteVectorStore> sqlite_store_;
  std::string table_name_;
  // Name of the column that stores the source text.
  std::string text_column_name_;
  // Name of the column that stores image bytes.
  std::string image_column_name_;
  // Name of the column that stores audio bytes.
  std::string audio_column_name_;
  // Name of the column that stores the embedding vector.
  std::string embedding_column_name_;
  // Dimension of the embedding vector.
  int embedding_dimension_;
};

}  // namespace retrieval
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_RETRIEVAL_SQLITE_MEMORY_STORE_H_
