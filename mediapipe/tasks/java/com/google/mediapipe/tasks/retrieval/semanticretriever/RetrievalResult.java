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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.mediapipe.tasks.core.Part;
import java.util.List;
import java.util.Map;

/** Represents a retrieved document or multimedia chunk. */
public final class RetrievalResult {
  private final String id;
  private final List<Part> content;
  private final Map<String, String> metadata;
  private final double score;

  /**
   * Creates a new RetrievalResult. Package-private to restrict construction to the retrieval
   * package.
   */
  RetrievalResult(String id, List<Part> content, Map<String, String> metadata, double score) {
    this.id = id;
    this.content = content != null ? ImmutableList.copyOf(content) : ImmutableList.of();
    this.metadata = metadata != null ? ImmutableMap.copyOf(metadata) : ImmutableMap.of();
    this.score = score;
  }

  /** Returns the unique identifier of the retrieved record. */
  public String id() {
    return id;
  }

  /** Returns the list of multi-modal parts in the retrieved record. */
  public List<Part> content() {
    return content;
  }

  /** Returns the metadata associated with the retrieved record. */
  public Map<String, String> metadata() {
    return metadata;
  }

  /** Returns the similarity/relevance score of this retrieved result. */
  public double score() {
    return score;
  }
}
