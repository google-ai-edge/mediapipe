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

import {computeCosineSimilarity} from '../../../../tasks/web/components/utils/cosine_similarity';
import {RetrievalRecord} from './retrieval_record';
import {VectorStore} from './vector_store';

function clonePart(
  part: RetrievalRecord['content'][number],
): RetrievalRecord['content'][number] {
  if ('imageBytes' in part) {
    return {
      ...part,
      imageBytes: new Uint8Array(part.imageBytes),
    };
  }
  if ('audioData' in part) {
    return {
      ...part,
      audioData: new Float32Array(part.audioData),
    };
  }
  return {...part};
}

function cloneRecord(record: RetrievalRecord): RetrievalRecord {
  return {
    ...record,
    content: record.content
      ? record.content.map((part) => clonePart(part))
      : [],
    embeddings: record.embeddings
      ? new Float32Array(record.embeddings)
      : new Float32Array(0),
    metadata: record.metadata ? {...record.metadata} : {},
    ...(record.childIds ? {childIds: [...record.childIds]} : {}),
  };
}

/**
 * Lightweight, in-memory implementation of VectorStore in TypeScript.
 */
export class MemoryVectorStore implements VectorStore {
  /**
   * 1:1 map from unique record ID to its `RetrievalRecord`.
   */
  private readonly records = new Map<string, RetrievalRecord>();

  upsert(records: readonly RetrievalRecord[]): void {
    for (const record of records) {
      this.records.set(record.id, cloneRecord(record));
    }
  }

  delete(ids: readonly string[]): void {
    const idSet = new Set(ids);
    for (const id of ids) {
      const existing = this.records.get(id);
      if (existing?.childIds) {
        for (const childId of existing.childIds) {
          idSet.add(childId);
        }
      }
    }
    for (const id of idSet) {
      this.records.delete(id);
    }
  }

  deleteByMetadata(metadataFilter: Record<string, string>): void {
    const filterKeys = Object.keys(metadataFilter);
    if (filterKeys.length === 0) {
      return;
    }
    const matchingIds: string[] = [];
    for (const record of this.records.values()) {
      if (record.parentId) continue;
      if (!record.metadata) continue;
      const matches = filterKeys.every(
        (key) => record.metadata![key] === metadataFilter[key],
      );
      if (matches) {
        matchingIds.push(record.id);
      }
    }
    if (matchingIds.length > 0) {
      this.delete(matchingIds);
    }
  }

  search(
    queryEmbedding: Float32Array | readonly number[],
    topK: number,
    metadataFilter?: Record<string, string>,
  ): RetrievalRecord[] {
    if (topK <= 0) {
      throw new Error(`'topK' must be greater than 0, got ${topK}.`);
    }
    const filterKeys = metadataFilter ? Object.keys(metadataFilter) : [];
    const queryEmbeddingArray = Array.from(queryEmbedding);

    const scored: Array<{record: RetrievalRecord; score: number}> = [];
    for (const record of this.records.values()) {
      if (metadataFilter && filterKeys.length > 0) {
        if (!record.metadata) continue;
        const matches = filterKeys.every(
          (key) => record.metadata?.[key] === metadataFilter[key],
        );
        if (!matches) continue;
      }
      if (!record.embeddings || record.embeddings.length === 0) continue;
      try {
        const score = computeCosineSimilarity(
          {floatEmbedding: queryEmbeddingArray, headIndex: 0, headName: ''},
          {
            floatEmbedding: Array.from(record.embeddings),
            headIndex: 0,
            headName: '',
          },
        );
        scored.push({record, score});
      } catch {
        // Skip invalid embeddings
      }
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map((item) => cloneRecord(item.record));
  }

  get(ids: readonly string[]): RetrievalRecord[] {
    const results: RetrievalRecord[] = [];
    for (const id of ids) {
      const record = this.records.get(id);
      if (record) {
        results.push(cloneRecord(record));
      }
    }
    return results;
  }

  getAllRecordIds(): string[] {
    const ids: string[] = [];
    for (const [id, record] of this.records.entries()) {
      if (!record.parentId) {
        ids.push(id);
      }
    }
    return ids;
  }

  deleteAll(): void {
    this.records.clear();
  }

  close(): void {
    this.records.clear();
  }
}
