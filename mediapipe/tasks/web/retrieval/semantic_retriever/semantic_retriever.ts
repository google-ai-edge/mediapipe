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
import type {
  ContentPart,
  EmbeddingProvider,
} from '../../../../tasks/web/core/embedding_provider';

import type {RetrievalOptions} from './retrieval_options';
import type {RetrievalRecord} from './retrieval_record';
import {SemanticRetrieverComponents} from './semantic_retriever_components';
import type {RetrievalResult} from './semantic_retriever_result';
import type {TextChunker} from './text_chunker';
import type {VectorStore} from './vector_store';

export {DefaultTextChunker} from './default_text_chunker';
export {MemoryVectorStore} from './memory_vector_store';
export type {RetrievalOptions} from './retrieval_options';
export type {RetrievalRecord} from './retrieval_record';
export {SemanticRetrieverComponents} from './semantic_retriever_components';
export type {RetrievalResult} from './semantic_retriever_result';
export type {ChunkingMode, TextChunker} from './text_chunker';
export type {VectorStore} from './vector_store';
export type {ContentPart, EmbeddingProvider};

/**
 * SemanticRetriever coordinates document chunking, multimodal embedding,
 * and high-dimensional vector search.
 */
export class SemanticRetriever {
  private isClosed = false;

  /**
   * Creates a SemanticRetriever from SemanticRetrieverComponents.
   *
   * @param components The SemanticRetriever components.
   * @return A Promise resolving to the SemanticRetriever instance.
   * @export
   */
  static async createFromComponents(
    components: SemanticRetrieverComponents,
  ): Promise<SemanticRetriever> {
    if (!components || components.providers().length === 0) {
      throw new Error(
        'At least one EmbeddingProvider must be registered in SemanticRetrieverComponents.',
      );
    }

    const vectorStore = components.vectorStore();
    if (!vectorStore) {
      throw new Error(
        'VectorStore must be provided in SemanticRetrieverComponents.',
      );
    }

    const textChunker = components.textChunker();
    if (!textChunker) {
      throw new Error(
        'TextChunker must be provided in SemanticRetrieverComponents.',
      );
    }

    return new SemanticRetriever(components, vectorStore, textChunker);
  }

  /** @hideconstructor */
  private constructor(
    private readonly components: SemanticRetrieverComponents,
    private readonly vectorStore: VectorStore,
    private readonly textChunker: TextChunker,
  ) {}

  private async embedContent(
    parts: readonly ContentPart[],
  ): Promise<Float32Array> {
    for (const provider of this.components.providers()) {
      const result = await provider.embedContent(parts);
      if (result != null) {
        return result instanceof Float32Array
          ? result
          : new Float32Array(result);
      }
    }
    throw new Error('No embedding provider found that can embed this content.');
  }

  /**
   * Inserts a text document into the retriever with automatic chunking.
   * @export
   */
  async insertDocument(
    id: string,
    text: string,
    metadata?: Record<string, string>,
  ): Promise<void> {
    this.ensureNotClosed();
    return this.insertContent(id, [{text}], metadata);
  }

  /**
   * Inserts multimodal content parts under a single unique ID.
   * @export
   */
  async insertContent(
    id: string,
    parts: readonly ContentPart[],
    metadata?: Record<string, string>,
  ): Promise<void> {
    this.ensureNotClosed();
    await this.vectorStore.delete([id]);

    const databaseRecords: Array<{
      id: string;
      parts: readonly ContentPart[];
      parentId?: string;
      childIds?: string[];
    }> = [];
    let childCount = 0;

    const isSingleTextPart =
      parts.length === 1 &&
      'text' in parts[0] &&
      typeof parts[0].text === 'string';

    if (isSingleTextPart) {
      const textPart = parts[0] as {text: string};
      const chunks = await this.textChunker.chunk(textPart.text);
      if (chunks.length > 1) {
        const childIds: string[] = [];
        for (const chunkText of chunks) {
          const childId = `${id}_chunk_${childCount++}`;
          childIds.push(childId);
          databaseRecords.push({
            id: childId,
            parts: [{text: chunkText}],
            parentId: id,
          });
        }
        databaseRecords.push({
          id,
          parts,
          childIds,
        });
      } else {
        databaseRecords.push({id, parts});
      }
    } else {
      databaseRecords.push({id, parts});
    }

    const recordsToUpsert: RetrievalRecord[] = [];
    for (const record of databaseRecords) {
      const isParentWithChildren =
        record.childIds !== undefined && record.childIds.length > 0;
      const embedding = isParentWithChildren
        ? new Float32Array(0)
        : await this.embedContent(record.parts);
      recordsToUpsert.push({
        id: record.id,
        content: [...record.parts],
        embeddings: embedding,
        metadata: metadata || {},
        parentId: record.parentId,
        childIds: record.childIds,
      });
    }
    await this.vectorStore.upsert(recordsToUpsert);
  }

  /**
   * Inserts an image into the retriever.
   * @export
   */
  async insertImage(
    id: string,
    imageBytes: Uint8Array,
    metadata?: Record<string, string>,
  ): Promise<void> {
    this.ensureNotClosed();
    return this.insertContent(id, [{imageBytes}], metadata);
  }

  /**
   * Inserts pre-decoded PCM float audio samples into the retriever.
   * @export
   */
  async insertAudio(
    id: string,
    audioData: Float32Array,
    metadata?: Record<string, string>,
  ): Promise<void> {
    this.ensureNotClosed();
    return this.insertContent(id, [{audioData}], metadata);
  }

  /**
   * Retrieves top-K nearest matching records based on a text or multimodal query.
   * @export
   */
  async retrieve(
    query: string | readonly ContentPart[],
    options: RetrievalOptions = {},
  ): Promise<RetrievalResult[]> {
    this.ensureNotClosed();
    const {limit = 5, metadataFilter} = options;
    if (limit <= 0) {
      throw new Error(`'limit' must be greater than 0, got ${limit}.`);
    }
    const queryParts: readonly ContentPart[] =
      typeof query === 'string' ? [{text: query}] : query;
    const queryEmbedding = await this.embedContent(queryParts);
    // Oversample candidates to avoid under-fetching during parent deduplication.
    const queryLimit = limit * 3;
    const searchResults = await this.vectorStore.search(
      queryEmbedding,
      queryLimit,
      metadataFilter,
    );

    const uniqueParentIds = new Set<string>();
    for (const record of searchResults) {
      if (record.parentId) {
        uniqueParentIds.add(record.parentId);
      }
    }

    const parentRecordsMap = new Map<string, RetrievalRecord>();
    if (uniqueParentIds.size > 0) {
      const parentRecords = await this.vectorStore.get(
        Array.from(uniqueParentIds),
      );
      for (const parent of parentRecords) {
        parentRecordsMap.set(parent.id, parent);
      }
    }

    const deduplicatedResults: RetrievalRecord[] = [];
    const seenParents = new Set<string>();
    for (const record of searchResults) {
      if (record.parentId) {
        if (!seenParents.has(record.parentId)) {
          seenParents.add(record.parentId);
          const parent = parentRecordsMap.get(record.parentId);
          if (parent) {
            deduplicatedResults.push({
              id: parent.id,
              content: parent.content,
              embeddings: record.embeddings,
              metadata: parent.metadata,
              parentId: parent.parentId,
              childIds: parent.childIds,
            });
          }
        }
      } else {
        if (!seenParents.has(record.id)) {
          seenParents.add(record.id);
          deduplicatedResults.push(record);
        }
      }
    }

    const clientResults: RetrievalResult[] = [];
    const queryEmbeddingArray = Array.from(queryEmbedding);
    for (const record of deduplicatedResults) {
      let score = 0;
      if (record.embeddings && record.embeddings.length > 0) {
        try {
          score = computeCosineSimilarity(
            {
              floatEmbedding: queryEmbeddingArray,
              headIndex: 0,
              headName: '',
            },
            {
              floatEmbedding: Array.from(record.embeddings),
              headIndex: 0,
              headName: '',
            },
          );
        } catch {
          score = 0;
        }
      }
      clientResults.push({
        id: record.id,
        content: record.content ? [...record.content] : [],
        metadata: record.metadata ? {...record.metadata} : {},
        score,
      });
      if (clientResults.length >= limit) {
        break;
      }
    }
    return clientResults;
  }

  /**
   * Deletes records and their chunks by ID(s), or records matching metadata filter.
   * @export
   */
  async delete(
    idsOrFilter: string | readonly string[] | Record<string, string>,
  ): Promise<void> {
    this.ensureNotClosed();
    if (typeof idsOrFilter === 'string') {
      await this.vectorStore.delete([idsOrFilter]);
    } else if (Array.isArray(idsOrFilter)) {
      await this.vectorStore.delete(idsOrFilter);
    } else {
      await this.vectorStore.deleteByMetadata(
        idsOrFilter as Record<string, string>,
      );
    }
  }

  /**
   * Retrieves all unique top-level record IDs stored in the retriever.
   * @export
   */
  async getAllRecordIds(): Promise<string[]> {
    this.ensureNotClosed();
    const ids = await this.vectorStore.getAllRecordIds();
    return Array.from(new Set(ids));
  }

  /**
   * Deletes all records from the retriever.
   * @export
   */
  async deleteAll(): Promise<void> {
    this.ensureNotClosed();
    await this.vectorStore.deleteAll();
  }

  /**
   * Closes the retriever and frees resources.
   * @export
   */
  close(): void {
    if (!this.isClosed) {
      this.isClosed = true;
      try {
        this.vectorStore.close();
      } catch {
        // Ignore closing errors.
      }
    }
  }

  private ensureNotClosed(): void {
    if (this.isClosed) {
      throw new Error('SemanticRetriever has already been closed.');
    }
  }
}


