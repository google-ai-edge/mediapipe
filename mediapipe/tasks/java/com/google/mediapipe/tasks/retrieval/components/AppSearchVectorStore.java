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

import static com.google.common.base.Strings.isNullOrEmpty;

import android.content.Context;
import android.net.Uri;
import androidx.appsearch.app.AppSearchBatchResult;
import androidx.appsearch.app.AppSearchSchema;
import androidx.appsearch.app.AppSearchSession;
import androidx.appsearch.app.EmbeddingVector;
import androidx.appsearch.app.GenericDocument;
import androidx.appsearch.app.GetByDocumentIdRequest;
import androidx.appsearch.app.PutDocumentsRequest;
import androidx.appsearch.app.RemoveByDocumentIdRequest;
import androidx.appsearch.app.SearchResult;
import androidx.appsearch.app.SearchResults;
import androidx.appsearch.app.SearchSpec;
import androidx.appsearch.app.SetSchemaRequest;
import androidx.appsearch.localstorage.LocalStorage;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.tasks.core.Part;
import com.google.mediapipe.tasks.retrieval.model.AudioPart;
import com.google.mediapipe.tasks.retrieval.model.ImagePart;
import com.google.mediapipe.tasks.retrieval.model.RetrievalRecord;
import com.google.mediapipe.tasks.text.core.TextPart;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/** A vector store implementation using Android system-level AppSearch storage. */
public final class AppSearchVectorStore implements VectorStore {
  private static final String SCHEMA_TYPE = "VectorStoreRecord";
  private static final String NAMESPACE = "mediapipe";
  private static final String MODEL_SIGNATURE = "mediapipe_retrieval_v1";

  private final AppSearchSession session;

  public AppSearchVectorStore(Context context, String databaseName) {
    try {
      String dbName = context.getPackageName() + "_" + databaseName;
      this.session =
          LocalStorage.createSearchSessionAsync(
                  new LocalStorage.SearchContext.Builder(context, dbName).build())
              .get();
      setSchema();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "AppSearch initialization was interrupted: " + e.getMessage(),
          e);
    } catch (Exception e) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "Failed to initialize AppSearchSession: " + e.getMessage(),
          e);
    }
  }

  private void setSchema() throws Exception {
    AppSearchSchema schema =
        new AppSearchSchema.Builder(SCHEMA_TYPE)
            .addProperty(
                new AppSearchSchema.StringPropertyConfig.Builder("contentType")
                    .setCardinality(AppSearchSchema.PropertyConfig.CARDINALITY_OPTIONAL)
                    .build())
            .addProperty(
                new AppSearchSchema.StringPropertyConfig.Builder("data")
                    .setCardinality(AppSearchSchema.PropertyConfig.CARDINALITY_OPTIONAL)
                    .build())
            .addProperty(
                new AppSearchSchema.StringPropertyConfig.Builder("metadata")
                    .setCardinality(AppSearchSchema.PropertyConfig.CARDINALITY_OPTIONAL)
                    .build())
            .addProperty(
                new AppSearchSchema.StringPropertyConfig.Builder("parentId")
                    .setCardinality(AppSearchSchema.PropertyConfig.CARDINALITY_OPTIONAL)
                    .build())
            .addProperty(
                new AppSearchSchema.StringPropertyConfig.Builder("childIds")
                    .setCardinality(AppSearchSchema.PropertyConfig.CARDINALITY_OPTIONAL)
                    .build())
            .addProperty(
                new AppSearchSchema.EmbeddingPropertyConfig.Builder("embeddings")
                    .setCardinality(AppSearchSchema.PropertyConfig.CARDINALITY_OPTIONAL)
                    .setIndexingType(
                        AppSearchSchema.EmbeddingPropertyConfig
                            .INDEXING_TYPE_APPROXIMATE_NEAREST_NEIGHBOR)
                    .build())
            .build();

    SetSchemaRequest setSchemaRequest =
        new SetSchemaRequest.Builder().addSchemas(schema).setForceOverride(true).build();
    session.setSchemaAsync(setSchemaRequest).get();
  }

  @Override
  public void upsert(List<RetrievalRecord> records) {
    if (records.isEmpty()) {
      return;
    }
    List<GenericDocument> docs = new ArrayList<>();
    for (RetrievalRecord record : records) {
      String contentType = "TEXT";
      String payload = "";
      if (!record.getContent().isEmpty()) {
        Part part = record.getContent().get(0);
        if (part instanceof TextPart) {
          contentType = "TEXT";
          payload = ((TextPart) part).getText();
        } else if (part instanceof ImagePart) {
          contentType = "IMAGE";
          payload = ((ImagePart) part).filePath().toString();
        } else if (part instanceof AudioPart) {
          contentType = "AUDIO";
          payload = ((AudioPart) part).filePath().toString();
        } else {
          throw new IllegalArgumentException("Unsupported part type");
        }
      }

      String serializedMetadata = new JSONObject(record.getMetadata()).toString();

      GenericDocument.Builder<?> docBuilder =
          new GenericDocument.Builder<>(NAMESPACE, record.getId(), SCHEMA_TYPE)
              .setPropertyString("contentType", contentType)
              .setPropertyString("data", payload)
              .setPropertyString("metadata", serializedMetadata)
              .setPropertyEmbedding(
                  "embeddings", new EmbeddingVector(record.getEmbeddings(), MODEL_SIGNATURE));

      if (record.getParentId() != null) {
        docBuilder.setPropertyString("parentId", record.getParentId());
      }
      if (!record.getChildIds().isEmpty()) {
        docBuilder.setPropertyString("childIds", new JSONArray(record.getChildIds()).toString());
      }

      docs.add(docBuilder.build());
    }

    try {
      PutDocumentsRequest putDocumentsRequest =
          new PutDocumentsRequest.Builder().addGenericDocuments(docs).build();
      session.putAsync(putDocumentsRequest).get();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "AppSearch upsert was interrupted: " + e.getMessage(),
          e);
    } catch (Exception e) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "Failed to upsert records into AppSearch: " + e.getMessage(),
          e);
    }
  }

  @Override
  public void delete(List<String> ids) {
    if (ids.isEmpty()) {
      return;
    }
    try {
      List<RetrievalRecord> records = get(ids);
      List<String> allIdsToDelete = new ArrayList<>(ids);
      for (RetrievalRecord record : records) {
        if (record.getChildIds() != null && !record.getChildIds().isEmpty()) {
          allIdsToDelete.addAll(record.getChildIds());
        }
      }
      RemoveByDocumentIdRequest deleteRequest =
          new RemoveByDocumentIdRequest.Builder(NAMESPACE).addIds(allIdsToDelete).build();
      session.removeAsync(deleteRequest).get();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "AppSearch delete was interrupted: " + e.getMessage(),
          e);
    } catch (Exception e) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "Failed to delete records from AppSearch: " + e.getMessage(),
          e);
    }
  }

  @Override
  public void delete(Map<String, String> metadataFilter) {
    if (metadataFilter == null || metadataFilter.isEmpty()) {
      return;
    }
    SearchSpec searchSpec = new SearchSpec.Builder().setResultCountPerPage(10000).build();
    try (SearchResults results = session.search("", searchSpec)) {
      List<String> idsToDelete = new ArrayList<>();
      List<SearchResult> page = results.getNextPageAsync().get();
      while (page != null && !page.isEmpty()) {
        for (SearchResult result : page) {
          RetrievalRecord record = toRetrievalRecord(result.getGenericDocument());
          if (matchesMetadataFilter(record, metadataFilter)) {
            idsToDelete.add(record.getId());
          }
        }
        page = results.getNextPageAsync().get();
      }
      if (!idsToDelete.isEmpty()) {
        delete(idsToDelete);
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "AppSearch delete was interrupted: " + e.getMessage(),
          e);
    } catch (Exception e) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "Failed to delete records from AppSearch: " + e.getMessage(),
          e);
    }
  }

  @Override
  public List<String> getAllRecordIds() {
    SearchSpec searchSpec = new SearchSpec.Builder().setResultCountPerPage(10000).build();
    try (SearchResults results = session.search("", searchSpec)) {
      Set<String> idSet = new LinkedHashSet<>();
      List<SearchResult> page = results.getNextPageAsync().get();
      while (page != null && !page.isEmpty()) {
        for (SearchResult result : page) {
          GenericDocument doc = result.getGenericDocument();
          String parentId = doc.getPropertyString("parentId");
          if (isNullOrEmpty(parentId)) {
            idSet.add(doc.getId());
          }
        }
        page = results.getNextPageAsync().get();
      }
      return new ArrayList<>(idSet);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("Interrupted while getting all record IDs", e);
    } catch (Exception e) {
      throw new IllegalStateException("Failed to get all record IDs", e);
    }
  }

  @Override
  public void deleteAll() {
    try {
      List<String> ids = getAllRecordIds();
      delete(ids);
    } catch (RuntimeException e) {
      throw new IllegalStateException("Failed to delete all records", e);
    }
  }

  private RetrievalRecord toRetrievalRecord(GenericDocument doc) {
    String id = doc.getId();
    String contentType = doc.getPropertyString("contentType");
    String data = doc.getPropertyString("data");
    String serializedMetadata = doc.getPropertyString("metadata");

    Map<String, String> metadata = new HashMap<>();
    if (serializedMetadata != null && !serializedMetadata.isEmpty()) {
      try {
        JSONObject json = new JSONObject(serializedMetadata);
        Iterator<String> keys = json.keys();
        while (keys.hasNext()) {
          String key = keys.next();
          metadata.put(key, json.getString(key));
        }
      } catch (JSONException e) {
        // Ignore or log
      }
    }

    String parentId = doc.getPropertyString("parentId");
    String childIdsStr = doc.getPropertyString("childIds");
    List<String> childIds = null;
    if (childIdsStr != null && !childIdsStr.isEmpty()) {
      if (childIdsStr.startsWith("[") && childIdsStr.endsWith("]")) {
        try {
          JSONArray jsonArray = new JSONArray(childIdsStr);
          childIds = new ArrayList<>();
          for (int i = 0; i < jsonArray.length(); i++) {
            childIds.add(jsonArray.getString(i));
          }
        } catch (JSONException e) {
          childIds = Arrays.asList(childIdsStr.split(","));
        }
      } else {
        childIds = Arrays.asList(childIdsStr.split(","));
      }
    }

    EmbeddingVector[] embeddingVectors = doc.getPropertyEmbeddingArray("embeddings");
    float[] embeddings = null;
    if (embeddingVectors != null && embeddingVectors.length > 0) {
      embeddings = embeddingVectors[0].getValues();
    }

    if (contentType == null) {
      contentType = "TEXT";
    }

    List<Part> content = new ArrayList<>();
    switch (contentType) {
      case "IMAGE":
        content.add(new ImagePart(Uri.parse(data)));
        break;
      case "AUDIO":
        content.add(new AudioPart(Uri.parse(data)));
        break;
      case "TEXT":
      default:
        content.add(new TextPart(data));
        break;
    }

    return new RetrievalRecord(id, content, embeddings, metadata, parentId, childIds);
  }

  @Override
  public List<RetrievalRecord> search(float[] queryEmbedding, int topK) {
    try {
      SearchSpec searchSpec =
          new SearchSpec.Builder()
              .setListFilterQueryLanguageEnabled(true)
              .addFilterNamespaces(NAMESPACE)
              .addFilterSchemas(SCHEMA_TYPE)
              .setResultCountPerPage(topK)
              .addEmbeddingParameters(new EmbeddingVector(queryEmbedding, MODEL_SIGNATURE))
              .setDefaultEmbeddingSearchMetricType(SearchSpec.EMBEDDING_SEARCH_METRIC_TYPE_COSINE)
              .setRankingStrategy(
                  "maxOrDefault(this.matchedSemanticScores(getEmbeddingParameter(0)), 0)")
              .build();

      SearchResults searchResults =
          session.search("semanticSearch(getEmbeddingParameter(0))", searchSpec);
      List<SearchResult> page = searchResults.getNextPageAsync().get();

      List<RetrievalRecord> records = new ArrayList<>();
      for (SearchResult result : page) {
        records.add(toRetrievalRecord(result.getGenericDocument()));
      }
      return records;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "AppSearch search was interrupted: " + e.getMessage(),
          e);
    } catch (Exception e) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "Failed to search AppSearch: " + e.getMessage(),
          e);
    }
  }

  @Override
  public List<RetrievalRecord> search(
      float[] queryEmbedding, int topK, Map<String, String> metadataFilter) {
    if (metadataFilter == null || metadataFilter.isEmpty()) {
      return search(queryEmbedding, topK);
    }
    try {
      int pageSize = Math.max(topK, 50);
      SearchSpec searchSpec =
          new SearchSpec.Builder()
              .setListFilterQueryLanguageEnabled(true)
              .addFilterNamespaces(NAMESPACE)
              .addFilterSchemas(SCHEMA_TYPE)
              .setResultCountPerPage(pageSize)
              .addEmbeddingParameters(new EmbeddingVector(queryEmbedding, MODEL_SIGNATURE))
              .setDefaultEmbeddingSearchMetricType(SearchSpec.EMBEDDING_SEARCH_METRIC_TYPE_COSINE)
              .setRankingStrategy(
                  "maxOrDefault(this.matchedSemanticScores(getEmbeddingParameter(0)), 0)")
              .build();

      SearchResults searchResults =
          session.search("semanticSearch(getEmbeddingParameter(0))", searchSpec);
      List<SearchResult> page = searchResults.getNextPageAsync().get();

      List<RetrievalRecord> records = new ArrayList<>();
      while (page != null && !page.isEmpty() && records.size() < topK) {
        for (SearchResult result : page) {
          RetrievalRecord record = toRetrievalRecord(result.getGenericDocument());
          if (matchesMetadataFilter(record, metadataFilter)) {
            records.add(record);
            if (records.size() == topK) {
              break;
            }
          }
        }
        if (records.size() < topK) {
          page = searchResults.getNextPageAsync().get();
        }
      }
      return records;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "AppSearch search was interrupted: " + e.getMessage(),
          e);
    } catch (Exception e) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "Failed to search AppSearch: " + e.getMessage(),
          e);
    }
  }

  private static boolean matchesMetadataFilter(
      RetrievalRecord record, Map<String, String> metadataFilter) {
    if (metadataFilter == null || metadataFilter.isEmpty()) {
      return true;
    }
    Map<String, String> metadata = record.getMetadata();
    if (metadata == null) {
      return false;
    }
    for (Map.Entry<String, String> entry : metadataFilter.entrySet()) {
      if (!Objects.equals(entry.getValue(), metadata.get(entry.getKey()))) {
        return false;
      }
    }
    return true;
  }

  @Override
  public List<RetrievalRecord> get(List<String> ids) {
    if (ids.isEmpty()) {
      return new ArrayList<>();
    }
    try {
      GetByDocumentIdRequest getRequest =
          new GetByDocumentIdRequest.Builder(NAMESPACE).addIds(ids).build();
      AppSearchBatchResult<String, GenericDocument> resultsBatch =
          session.getByDocumentIdAsync(getRequest).get();
      Map<String, GenericDocument> resultsMap = resultsBatch.getSuccesses();
      List<RetrievalRecord> records = new ArrayList<>();
      for (String id : ids) {
        GenericDocument doc = resultsMap.get(id);
        if (doc == null) {
          continue;
        }
        records.add(toRetrievalRecord(doc));
      }
      return records;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "AppSearch get was interrupted: " + e.getMessage(),
          e);
    } catch (Exception e) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL,
          "Failed to get records from AppSearch: " + e.getMessage(),
          e);
    }
  }

  @Override
  public void close() {
    if (session != null) {
      session.close();
    }
  }
}
