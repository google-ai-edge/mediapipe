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

package com.google.mediapipe.tasks.retrieval.components;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.util.List;
import java.util.Map;

/** The memory record entity for vector stores. */
public final class VectorStoreRecord<T> {
  /** The content types supported by VectorStoreRecord. */
  public enum ContentType {
    TEXT,
    IMAGE,
    AUDIO,
    UNKNOWN;
  }

  private final T data;
  private final ImmutableList<Float> embeddings;
  private final ImmutableMap<String, Object> metadata;
  private final ContentType contentType;
  private final String recordId;

  private VectorStoreRecord(
      String recordId,
      ContentType contentType,
      T data,
      List<Float> embeddings,
      Map<String, Object> metadata) {
    this.recordId = recordId;
    this.contentType = contentType;
    this.data = data;
    this.embeddings = ImmutableList.copyOf(embeddings);
    this.metadata = ImmutableMap.copyOf(metadata);
  }

  public T getData() {
    return data;
  }

  public ImmutableList<Float> getEmbeddings() {
    return embeddings;
  }

  public ImmutableMap<String, Object> getMetadata() {
    return metadata;
  }

  public ContentType getContentType() {
    return contentType;
  }

  public String getRecordId() {
    return recordId;
  }

  public static <T> VectorStoreRecord<T> create(
      String recordId,
      ContentType contentType,
      T data,
      List<Float> embeddings,
      Map<String, Object> metadata) {
    return new VectorStoreRecord<T>(recordId, contentType, data, embeddings, metadata);
  }
}
