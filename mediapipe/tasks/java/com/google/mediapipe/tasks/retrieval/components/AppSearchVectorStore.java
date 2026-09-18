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

import android.content.Context;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.tasks.retrieval.model.RetrievalRecord;
import java.util.List;
import java.util.Map;

/** A vector store implementation using Android system-level AppSearch storage. */
public final class AppSearchVectorStore implements VectorStore {

  private final VectorStore impl;

  public AppSearchVectorStore(Context context, String databaseName) {
    try {
      // Check if AppSearch is available in the classpath before attempting to load the
      // implementation.
      Class.forName("androidx.appsearch.app.AppSearchSession");
      impl =
          Class.forName("com.google.mediapipe.tasks.retrieval.components.AppSearchVectorStoreImpl")
              .asSubclass(VectorStore.class)
              .getConstructor(Context.class, String.class)
              .newInstance(context, databaseName);
    } catch (ClassNotFoundException e) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION,
          "AppSearch dependencies are missing from the classpath. Please add"
              + " androidx.appsearch:appsearch and androidx.appsearch:appsearch-local-storage to"
              + " your build configuration to use AppSearchVectorStore.",
          e);
    } catch (ReflectiveOperationException e) {
      if (e.getCause() instanceof MediaPipeException) {
        throw (MediaPipeException) e.getCause();
      }
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "Failed to instantiate AppSearchVectorStoreImpl: " + e.getMessage(),
          e);
    }
  }

  @Override
  public void upsert(List<RetrievalRecord> records) {
    impl.upsert(records);
  }

  @Override
  public void delete(List<String> ids) {
    impl.delete(ids);
  }

  @Override
  public void delete(Map<String, String> metadataFilter) {
    impl.delete(metadataFilter);
  }

  @Override
  public List<String> getAllRecordIds() {
    return impl.getAllRecordIds();
  }

  @Override
  public void deleteAll() {
    impl.deleteAll();
  }

  @Override
  public List<RetrievalRecord> search(float[] queryEmbedding, int topK) {
    return impl.search(queryEmbedding, topK);
  }

  @Override
  public List<RetrievalRecord> search(
      float[] queryEmbedding, int topK, Map<String, String> metadataFilter) {
    return impl.search(queryEmbedding, topK, metadataFilter);
  }

  @Override
  public List<RetrievalRecord> get(List<String> ids) {
    return impl.get(ids);
  }

  @Override
  public void close() throws Exception {
    impl.close();
  }
}
