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
import com.google.common.base.Preconditions;
import com.google.mediapipe.tasks.retrieval.model.RetrievalRecord;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** A vector store implementation backed by SQLite database using native SQLite vector engine. */
public final class SqliteVectorStore implements VectorStore {
  private final String dbName;
  private final Context context;
  private long jniHandle = 0;
  private int dimension = 0;

  static {
    System.loadLibrary("sqlite_vector_store_jni");
  }

  /**
   * Initializes the SqliteVectorStore.
   *
   * @param context the Android context.
   * @param dbName the name of the database file.
   */
  public SqliteVectorStore(Context context, String dbName, int embeddingDimension) {
    Preconditions.checkArgument(embeddingDimension > 0, "Dimension must be greater than 0");
    this.context = context;
    File dbFile = context.getDatabasePath(dbName);
    File parentFile = dbFile.getParentFile();
    if (parentFile != null && !parentFile.exists()) {
      parentFile.mkdirs();
    }
    this.dbName = dbFile.getAbsolutePath();
    this.dimension = embeddingDimension;
    this.jniHandle = nativeCreateSqliteVectorStore(this.dimension, this.dbName);
    if (this.jniHandle == 0) {
      throw new IllegalStateException("Failed to initialize SqliteVectorStore.");
    }
  }

  private float[] resizeVector(float[] vec, int targetDim) {
    if (vec == null || vec.length == 0) {
      return new float[0];
    }
    float[] resized = new float[targetDim];
    if (vec.length <= targetDim) {
      System.arraycopy(vec, 0, resized, 0, vec.length);
    } else {
      System.arraycopy(vec, 0, resized, 0, targetDim);
    }
    return resized;
  }

  @Override
  public void upsert(List<RetrievalRecord> records) {
    if (records.isEmpty()) {
      return;
    }

    for (RetrievalRecord record : records) {
      float[] resizedEmbeddings = resizeVector(record.getEmbeddings(), this.dimension);
      RetrievalRecord recordWithResizedEmbeddings =
          new RetrievalRecord(
              record.getId(),
              record.getContent(),
              resizedEmbeddings,
              record.getMetadata(),
              record.getParentId(),
              record.getChildIds());
      nativeInsert(
          jniHandle, MemoryRecordConverter.toMemoryRecordProtoBytes(recordWithResizedEmbeddings));
    }
  }

  @Override
  public void delete(List<String> ids) {
    if (ids.isEmpty()) {
      return;
    }
    nativeDeleteRecords(jniHandle, ids.toArray(new String[0]));
  }

  @Override
  public void delete(Map<String, String> metadataFilter) {
    validateMetadataFilter(metadataFilter);
    if (metadataFilter == null || metadataFilter.isEmpty()) {
      return;
    }
    String[] filterKeys = new String[metadataFilter.size()];
    String[] filterValues = new String[metadataFilter.size()];
    int idx = 0;
    for (Map.Entry<String, String> entry : metadataFilter.entrySet()) {
      filterKeys[idx] = entry.getKey();
      filterValues[idx] = entry.getValue();
      ++idx;
    }
    nativeDeleteByMetadata(jniHandle, filterKeys, filterValues);
  }

  private static void validateMetadataFilter(Map<String, String> metadataFilter) {
    if (metadataFilter != null) {
      for (Map.Entry<String, String> entry : metadataFilter.entrySet()) {
        if (entry.getKey() == null || entry.getValue() == null) {
          throw new IllegalArgumentException(
              "metadataFilter cannot contain null keys or null values");
        }
      }
    }
  }

  @Override
  public List<String> getAllRecordIds() {
    return nativeGetAllRecordIds(jniHandle);
  }

  @Override
  public void deleteAll() {
    nativeSqlQuery(jniHandle, "DELETE FROM rag_vector_store");
  }

  private RetrievalRecord toRetrievalRecord(byte[] storeRecordBytes) {
    return MemoryRecordConverter.toRetrievalRecord(storeRecordBytes);
  }

  @Override
  public List<RetrievalRecord> search(float[] queryEmbedding, int topK) {
    return search(queryEmbedding, topK, /* metadataFilter= */ null);
  }

  @Override
  public List<RetrievalRecord> search(
      float[] queryEmbedding, int topK, Map<String, String> metadataFilter) {
    validateMetadataFilter(metadataFilter);
    if (queryEmbedding.length <= 0) {
      throw new IllegalArgumentException("Query embedding dimension must be greater than 0");
    }
    float[] resizedQuery = resizeVector(queryEmbedding, this.dimension);

    if (metadataFilter == null || metadataFilter.isEmpty()) {
      List<byte[]> results = nativeGetNearestRecords(jniHandle, resizedQuery, topK, 0.0f);
      List<RetrievalRecord> records = new ArrayList<>();
      for (byte[] storeRecordBytes : results) {
        records.add(toRetrievalRecord(storeRecordBytes));
      }
      return records;
    }

    String[] filterKeys = new String[metadataFilter.size()];
    String[] filterValues = new String[metadataFilter.size()];
    int idx = 0;
    for (Map.Entry<String, String> entry : metadataFilter.entrySet()) {
      filterKeys[idx] = entry.getKey();
      filterValues[idx] = entry.getValue();
      ++idx;
    }

    List<byte[]> results =
        nativeGetNearestRecordsWithFilter(
            jniHandle, resizedQuery, topK, 0.0f, filterKeys, filterValues);
    List<RetrievalRecord> records = new ArrayList<>();
    for (byte[] storeRecordBytes : results) {
      records.add(toRetrievalRecord(storeRecordBytes));
    }
    return records;
  }

  @Override
  public List<RetrievalRecord> get(List<String> ids) {
    if (ids.isEmpty()) {
      return new ArrayList<>();
    }
    List<byte[]> results = nativeGetRecords(jniHandle, ids.toArray(new String[0]));
    List<RetrievalRecord> records = new ArrayList<>();
    for (byte[] storeRecordBytes : results) {
      records.add(toRetrievalRecord(storeRecordBytes));
    }
    return records;
  }

  @Override
  public synchronized void close() {
    if (jniHandle != 0) {
      nativeClose(jniHandle);
      jniHandle = 0;
    }
  }
  private static native long nativeCreateSqliteVectorStore(
      int numEmbeddingDimensions, String databasePath);

  private static native void nativeInsert(long jniHandle, byte[] memoryRecordBytes);

  private static native List<byte[]> nativeGetNearestRecords(
      long jniHandle, float[] queryEmbeddings, int topK, float minSimilarityScore);

  private static native List<byte[]> nativeGetNearestRecordsWithFilter(
      long jniHandle,
      float[] queryEmbeddings,
      int topK,
      float minSimilarityScore,
      String[] filterKeys,
      String[] filterValues);

  private static native List<byte[]> nativeGetRecords(long jniHandle, String[] ids);

  private static native void nativeDeleteByMetadata(
      long jniHandle, String[] filterKeys, String[] filterValues);

  private static native List<String> nativeGetAllRecordIds(long jniHandle);

  private static native void nativeDeleteRecords(long jniHandle, String[] ids);

  private static native void nativeSqlQuery(long jniHandle, String query);

  private static native void nativeClose(long jniHandle);
}
