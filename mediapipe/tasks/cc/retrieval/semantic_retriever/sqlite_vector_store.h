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

#ifndef MEDIAPIPE_TASKS_CC_RETRIEVAL_SQLITE_VECTOR_STORE_H_
#define MEDIAPIPE_TASKS_CC_RETRIEVAL_SQLITE_VECTOR_STORE_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sqlite_helpers {
class Database;
class Statement;
}  // namespace sqlite_helpers

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/proto/vector_stores.pb.h"

namespace mediapipe {
namespace tasks {
namespace retrieval {

// A sqlite vector store based on a dynamically-loadable extension that provides
// vector manipulation functions to sqlite.
class SqliteVectorStore {
 public:
  struct ColumnValue {
    ColumnValue(absl::string_view column_name, absl::string_view value)
        : column_name_(column_name), string_value_(value) {}
    ColumnValue(absl::string_view column_name, std::vector<float> float_vector)
        : column_name_(column_name), float_vector_(std::move(float_vector)) {}
    const std::string& column_name() const { return column_name_; }
    const std::optional<std::string>& string_value() const {
      return string_value_;
    }
    // Stored in a BLOB column. Note that the column storing the embeddings must
    // have type BLOB_F32xN, where N is the number of embedding dimensions.
    const std::optional<std::vector<float>>& float_vector() const {
      return float_vector_;
    }

   private:
    std::string column_name_;
    std::optional<std::string> string_value_;
    std::optional<std::vector<float>> float_vector_;
  };

  // Open the database from the given path. If it is empty, create a new
  // in-memory database.
  virtual absl::Status Initialize(const std::string& database_path);
  virtual absl::Status CreateTables(std::vector<TableConfig> table_configs);
  // Inserts or replaces columns in the table.
  virtual absl::Status InsertOrReplace(
      const std::string& table_name,
      const std::vector<ColumnValue>& column_values);
  // Inserts or replaces multiple rows in a single transaction.
  virtual absl::Status InsertOrReplaceBatch(
      const std::string& table_name,
      absl::Span<const std::vector<ColumnValue>> rows);
  // Queries data from the database. Returns the data as tables.
  virtual absl::StatusOr<std::vector<std::vector<ColumnValue>>> Get(
      const std::string& sql_query,
      std::function<absl::Status(sqlite_helpers::Statement& statement)>
          sql_bind_parameter_callback);
  // Executes a SQL query that does not return results.
  virtual absl::Status Execute(absl::string_view sql_query);
  virtual absl::Status Execute(
      absl::string_view sql_query,
      std::function<absl::Status(sqlite_helpers::Statement& statement)>
          sql_bind_parameter_callback);

  SqliteVectorStore();
  virtual ~SqliteVectorStore();
  SqliteVectorStore(const SqliteVectorStore&) = delete;
  SqliteVectorStore& operator=(const SqliteVectorStore&) = delete;

 private:
  absl::Status CreateTable(const TableConfig& table_config);
  absl::StatusOr<const TableConfig::ColumnConfig*> GetColumnConfig(
      absl::string_view table_name, absl::string_view column_name);
  std::string database_path_;
  std::unique_ptr<sqlite_helpers::Database> db_;
  absl::flat_hash_map<std::string, std::vector<TableConfig::ColumnConfig>>
      tables_;
};

}  // namespace retrieval
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_RETRIEVAL_SQLITE_VECTOR_STORE_H_
