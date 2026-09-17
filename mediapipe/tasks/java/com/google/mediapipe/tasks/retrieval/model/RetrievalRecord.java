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

package com.google.mediapipe.tasks.retrieval.model;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.mediapipe.tasks.core.Part;
import java.util.List;
import java.util.Map;

/** Represents a single multi-modal record stored in the vector store. */
public class RetrievalRecord {
  private final String id;
  private final List<Part> content;
  private final float[] embeddings;
  private final Map<String, String> metadata;
  private final String parentId;
  private final List<String> childIds;

  /**
   * Creates a new RetrievalRecord.
   *
   * @param id the unique identifier of the record.
   * @param content the list of multi-modal parts that make up the record content.
   * @param embeddings the vector embeddings representing the record.
   * @param metadata the key-value metadata associated with the record.
   * @param parentId the id of the parent document, if any.
   * @param childIds the ids of the child chunks, if any.
   */
  public RetrievalRecord(
      String id,
      List<Part> content,
      float[] embeddings,
      Map<String, String> metadata,
      String parentId,
      List<String> childIds) {
    this.id = id;
    this.content = content != null ? ImmutableList.copyOf(content) : ImmutableList.of();
    this.embeddings = embeddings != null ? embeddings.clone() : new float[0];
    this.metadata = metadata != null ? ImmutableMap.copyOf(metadata) : ImmutableMap.of();
    this.parentId = parentId;
    this.childIds = childIds != null ? ImmutableList.copyOf(childIds) : ImmutableList.of();
  }

  public RetrievalRecord(
      String id, List<Part> content, float[] embeddings, Map<String, String> metadata) {
    this(id, content, embeddings, metadata, null, null);
  }

  /** Returns the unique identifier of the record. */
  public String getId() {
    return id;
  }

  /** Returns the list of multi-modal parts in the record. */
  public List<Part> getContent() {
    return content;
  }

  /** Returns a copy of the vector embeddings. */
  public float[] getEmbeddings() {
    return embeddings.clone();
  }

  /** Returns the metadata associated with the record. */
  public Map<String, String> getMetadata() {
    return metadata;
  }

  /** Returns the parent document id, if any. */
  public String getParentId() {
    return parentId;
  }

  /** Returns the list of child chunk ids, if any. */
  public List<String> getChildIds() {
    return childIds;
  }
}
