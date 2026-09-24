/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {RetrievalRecord} from './retrieval_record';

/**
 * Interface for vector storage and similarity search backends.
 */
export interface VectorStore {
  /** Inserts or updates records in the vector store. */
  upsert(records: readonly RetrievalRecord[]): Promise<void> | void;

  /** Deletes records matching the specified IDs. */
  delete(ids: readonly string[]): Promise<void> | void;

  /** Deletes records matching the specified metadata filter. */
  deleteByMetadata(
    metadataFilter: Record<string, string>,
  ): Promise<void> | void;

  /** Searches top-K nearest records to the query embedding. */
  search(
    queryEmbedding: Float32Array | readonly number[],
    topK: number,
    metadataFilter?: Record<string, string>,
  ): Promise<RetrievalRecord[]> | RetrievalRecord[];

  /** Retrieves records matching the specified IDs by key. */
  get(ids: readonly string[]): Promise<RetrievalRecord[]> | RetrievalRecord[];

  /** Retrieves all unique record identifiers. */
  getAllRecordIds(): Promise<string[]> | string[];

  /** Deletes all records from the vector store. */
  deleteAll(): Promise<void> | void;

  /** Releases resources associated with the vector store. */
  close(): void;
}
