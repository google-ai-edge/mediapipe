/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

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

package com.google.mediapipe.tasks.retrieval.semanticretriever;

import com.google.mediapipe.tasks.core.EmbeddingProvider;
import com.google.mediapipe.tasks.retrieval.chunking.TextChunker;
import com.google.mediapipe.tasks.retrieval.components.VectorStore;
import java.util.ArrayList;
import java.util.List;

/** Components used to initialize the SemanticRetriever. */
public final class SemanticRetrieverComponents {
  private VectorStore vectorStore;
  private final List<EmbeddingProvider> providers = new ArrayList<>();
  private TextChunker textChunker;

  public SemanticRetrieverComponents() {}

  /**
   * Sets the vector store for the SemanticRetriever.
   *
   * @param vectorStore the vector store to set.
   * @return a SemanticRetrieverComponents instance.
   */
  public SemanticRetrieverComponents setVectorStore(VectorStore vectorStore) {
    this.vectorStore = vectorStore;
    return this;
  }

  /**
   * Returns the vector store for the SemanticRetriever.
   *
   * @return the vector store for the SemanticRetriever.
   */
  public VectorStore vectorStore() {
    return vectorStore;
  }

  /**
   * Adds an embedding provider to the SemanticRetriever.
   *
   * @param provider the embedding provider to add.
   * @return a SemanticRetrieverComponents instance.
   */
  public SemanticRetrieverComponents addProvider(EmbeddingProvider provider) {
    this.providers.add(provider);
    return this;
  }

  /**
   * Returns the embedding providers for the SemanticRetriever.
   *
   * @return the embedding providers for the SemanticRetriever.
   */
  public List<EmbeddingProvider> providers() {
    return providers;
  }

  /**
   * Sets the text chunker for the SemanticRetriever.
   *
   * @param textChunker the text chunker to set.
   * @return a SemanticRetrieverComponents instance.
   */
  public SemanticRetrieverComponents setTextChunker(TextChunker textChunker) {
    this.textChunker = textChunker;
    return this;
  }

  /**
   * Returns the text chunker for the SemanticRetriever.
   *
   * @return the text chunker for the SemanticRetriever.
   */
  public TextChunker textChunker() {
    return textChunker;
  }
}
