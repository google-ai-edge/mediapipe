/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.mediapipe.tasks.retrieval.components;

import com.google.mediapipe.tasks.retrieval.model.RetrievalRecord;
import java.util.List;
import java.util.Map;

/** Interface for a vector store. */
public interface VectorStore extends AutoCloseable {
  /** Upserts a list of records in the vector store. */
  void upsert(List<RetrievalRecord> records);

  /** Deletes the records matching the specified IDs. */
  void delete(List<String> ids);

  /** Deletes the records matching the specified metadata filter. */
  void delete(Map<String, String> metadataFilter);

  /** Searches the top-K nearest records to the query embedding. */
  List<RetrievalRecord> search(float[] queryEmbedding, int topK);

  /** Searches the top-K nearest records matching the query embedding and metadata filter. */
  List<RetrievalRecord> search(
      float[] queryEmbedding, int topK, Map<String, String> metadataFilter);

  /** Retrieves the records matching the specified IDs (Key-Value lookup). */
  List<RetrievalRecord> get(List<String> ids);

  /** Retrieves all unique record identifiers in the vector store. */
  List<String> getAllRecordIds();

  /** Deletes all records from the vector store. */
  void deleteAll();
}
