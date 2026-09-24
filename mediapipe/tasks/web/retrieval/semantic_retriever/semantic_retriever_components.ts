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

import {EmbeddingProvider} from '../../../../tasks/web/core/embedding_provider';
import {TextChunker} from '../../../../tasks/web/retrieval/semantic_retriever/text_chunker';
import {VectorStore} from '../../../../tasks/web/retrieval/semantic_retriever/vector_store';

/**
 * Components used to initialize the SemanticRetriever.
 */
export class SemanticRetrieverComponents {
  private readonly providerList: EmbeddingProvider[] = [];
  private vectorStoreInstance?: VectorStore;
  private textChunkerInstance?: TextChunker;

  /**
   * Adds an embedding provider to the SemanticRetriever.
   *
   * @param provider The embedding provider to add.
   * @return The SemanticRetrieverComponents instance for chaining.
   */
  addProvider(provider: EmbeddingProvider): this {
    this.providerList.push(provider);
    return this;
  }

  /**
   * Returns the registered embedding providers for the SemanticRetriever.
   *
   * @return A list of registered embedding providers.
   */
  providers(): readonly EmbeddingProvider[] {
    return [...this.providerList];
  }

  /**
   * Sets the vector store for the SemanticRetriever.
   *
   * @param vectorStore The vector store to set.
   * @return The SemanticRetrieverComponents instance for chaining.
   */
  setVectorStore(vectorStore: VectorStore): this {
    this.vectorStoreInstance = vectorStore;
    return this;
  }

  /**
   * Returns the vector store for the SemanticRetriever.
   *
   * @return The vector store for the SemanticRetriever.
   */
  vectorStore(): VectorStore | undefined {
    return this.vectorStoreInstance;
  }

  /**
   * Sets the text chunker for the SemanticRetriever.
   *
   * @param textChunker The text chunker to set.
   * @return The SemanticRetrieverComponents instance for chaining.
   */
  setTextChunker(textChunker: TextChunker): this {
    this.textChunkerInstance = textChunker;
    return this;
  }

  /**
   * Returns the text chunker for the SemanticRetriever.
   *
   * @return The text chunker for the SemanticRetriever.
   */
  textChunker(): TextChunker | undefined {
    return this.textChunkerInstance;
  }
}
