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

import android.content.Context;
import android.net.Uri;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.mediapipe.tasks.components.containers.AudioData;
import com.google.mediapipe.tasks.components.containers.Embedding;
import com.google.mediapipe.tasks.components.utils.CosineSimilarity;
import com.google.mediapipe.tasks.core.AudioPart;
import com.google.mediapipe.tasks.core.EmbeddingProvider;
import com.google.mediapipe.tasks.core.ImagePart;
import com.google.mediapipe.tasks.core.Part;
import com.google.mediapipe.tasks.core.TextPart;
import com.google.mediapipe.tasks.core.logging.TasksStatsLogger;
import com.google.mediapipe.tasks.core.logging.TasksStatsLoggerFactory;
import com.google.mediapipe.tasks.retrieval.chunking.DefaultTextChunker;
import com.google.mediapipe.tasks.retrieval.chunking.TextChunker;
import com.google.mediapipe.tasks.retrieval.model.RetrievalRecord;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/** MediaPipe SemanticRetriever API that manages vector embedding and retrieval. */
@SuppressWarnings({"IfChainToSwitch", "PatternMatchingInstanceof"})
public final class SemanticRetriever implements AutoCloseable {
  private static final int DEFAULT_CHUNK_SIZE = 512;
  private static final int DEFAULT_CHUNK_OVERLAP = 100;

  private final Context context;
  private final SemanticRetrieverComponents components;
  private final TasksStatsLogger statsLogger;
  private final AtomicLong syntheticTimestamp = new AtomicLong(0);
  private final TextChunker textChunker;

  /**
   * Represents a prepared retrieval record and its parent/child relationships prior to vector
   * embedding generation and database insertion.
   */
  private static final class DatabaseRecord {
    final String id;
    final Part part;
    final String parentId;
    final List<String> childIds;

    DatabaseRecord(String id, Part part, String parentId, List<String> childIds) {
      this.id = id;
      this.part = part;
      this.parentId = parentId;
      this.childIds = childIds;
    }
  }

  private SemanticRetriever(Context context, SemanticRetrieverComponents components) {
    this.context = context;
    this.components = components;
    this.statsLogger =
        TasksStatsLoggerFactory.create(context, "SemanticRetriever", /* taskRunningModeStr= */ "");
    this.statsLogger.logSessionStart();
    this.textChunker =
        components.textChunker() != null
            ? components.textChunker()
            : new DefaultTextChunker(
                DEFAULT_CHUNK_SIZE,
                DEFAULT_CHUNK_OVERLAP,
                DefaultTextChunker.ChunkingMode.CHARACTER);
  }

  /**
   * Creates a {@link SemanticRetriever} from {@link SemanticRetrieverComponents}.
   *
   * @param context the Android context.
   * @param components the SemanticRetriever components.
   * @return a SemanticRetriever instance.
   */
  public static SemanticRetriever createFromComponents(
      Context context, SemanticRetrieverComponents components) {
    return new SemanticRetriever(context, components);
  }

  /**
   * Inserts a document into the vector store.
   *
   * @param recordId the id of the record to insert.
   * @param text the text of the record to insert.
   */
  public void insertDocument(String recordId, String text) {
    insertDocument(recordId, text, ImmutableMap.of());
  }

  /**
   * Inserts a document into the vector store with metadata.
   *
   * @param recordId the id of the record to insert.
   * @param text the text of the record to insert.
   * @param metadata the metadata of the record to insert.
   */
  public void insertDocument(String recordId, String text, Map<String, String> metadata) {
    insertContent(recordId, ImmutableList.of(new TextPart(text)), metadata);
  }

  /** Inserts an image into the vector store using a Uri. */
  public void insertImage(String recordId, Uri uri) {
    insertImage(recordId, uri, ImmutableMap.of());
  }

  /** Inserts an image into the vector store using a Uri and metadata. */
  public void insertImage(String recordId, Uri uri, Map<String, String> metadata) {
    insertContent(recordId, ImmutableList.of(new ImagePart(uri)), metadata);
  }

  /** Inserts audio into the vector store using a Uri. */
  public void insertAudio(String recordId, Uri uri) {
    insertAudio(recordId, uri, ImmutableMap.of());
  }

  /** Inserts audio into the vector store using a Uri and metadata. */
  public void insertAudio(String recordId, Uri uri, Map<String, String> metadata) {
    insertContent(recordId, ImmutableList.of(new AudioPart(uri)), metadata);
  }

  /**
   * Inserts content into the vector store.
   *
   * @param recordId the id of the record to insert.
   * @param parts the parts to insert.
   */
  public void insertContent(String recordId, List<Part> parts) {
    insertContent(recordId, parts, ImmutableMap.of());
  }

  /**
   * Inserts content into the vector store with metadata.
   *
   * @param recordId the id of the record to insert.
   * @param parts the parts to insert.
   * @param metadata the metadata of the record to insert.
   */
  public void insertContent(String recordId, List<Part> parts, Map<String, String> metadata) {
    // Delete any existing parent record and its associated child chunk records to prevent
    // orphaning.
    delete(ImmutableList.of(recordId));

    List<DatabaseRecord> databaseRecords = new ArrayList<>();
    int childCount = 0;

    // Phase 1: Iterate through parts and prepare database records
    for (Part part : parts) {
      if (part instanceof TextPart) {
        TextPart textPart = (TextPart) part;
        List<String> chunks = textChunker.chunk(textPart.getText());
        if (chunks.size() > 1) {
          List<String> childIds = new ArrayList<>();
          for (String chunkText : chunks) {
            String childId = recordId + "_chunk_" + childCount++;
            childIds.add(childId);
            databaseRecords.add(
                new DatabaseRecord(childId, new TextPart(chunkText), recordId, null));
          }
          databaseRecords.add(new DatabaseRecord(recordId, textPart, null, childIds));
          continue;
        }
      }
      databaseRecords.add(new DatabaseRecord(recordId, part, null, null));
    }

    // Phase 2: Generate embeddings and insert into vector store
    for (DatabaseRecord record : databaseRecords) {
      float[] embedding = embedContent(ImmutableList.of(record.part));
      persistPart(record.id, record.part, embedding, metadata, record.parentId, record.childIds);
    }
  }

  private void persistPart(
      String recordId,
      Part part,
      float[] embeddings,
      Map<String, String> metadata,
      String parentId,
      List<String> childIds) {
    ImmutableList.Builder<RetrievalRecord> records = ImmutableList.builder();
    records.add(
        new RetrievalRecord(
            recordId, ImmutableList.of(part), embeddings, metadata, parentId, childIds));
    components.vectorStore().upsert(records.build());
  }

  private float[] embedContent(List<Part> content) {
    List<Object> resolvedContent = new ArrayList<>();
    for (Part part : content) {
      if (part instanceof ImagePart) {
        ImagePart imagePart = (ImagePart) part;
        byte[] bytes = imagePart.imageBytes();
        if (bytes == null && imagePart.getFilePath() != null) {
          bytes = ImageDecoder.getImageBytes(context, imagePart.getFilePath());
        }
        if (bytes == null) {
          throw new IllegalArgumentException(
              "Failed to load image bytes from ImagePart: " + imagePart.getFilePath());
        }
        resolvedContent.add(bytes);
      } else if (part instanceof AudioPart) {
        AudioPart audioPart = (AudioPart) part;
        if (audioPart.audioData() != null) {
          resolvedContent.add(audioPart.audioData());
        } else if (audioPart.getFilePath() != null) {
          AudioData audio = AudioDecoder.decodeAudio(context, audioPart.getFilePath());
          if (audio == null) {
            throw new IllegalArgumentException(
                "Failed to load audio from URI: " + audioPart.getFilePath());
          }
          resolvedContent.add(audio);
        } else {
          throw new IllegalArgumentException("AudioPart must contain audioData or filePath.");
        }
      } else if (part instanceof TextPart) {
        TextPart textPart = (TextPart) part;
        resolvedContent.add(textPart.getText());
      } else {
        resolvedContent.add(part);
      }
    }

    for (EmbeddingProvider provider : components.providers()) {
      float[] embedding = provider.embedContent(resolvedContent);
      if (embedding != null) {
        return embedding;
      }
    }
    throw new IllegalStateException("No embedding provider found that can embed this content.");
  }

  /**
   * Retrieves the topK records from the vector store for the given query text.
   *
   * @param queryText the query text to retrieve records for.
   * @param topK the number of records to retrieve.
   * @return the topK records from the vector store for the given query text.
   */
  public List<RetrievalResult> retrieve(String queryText, int topK) {
    return retrieve(queryText, topK, /* metadataFilter= */ null);
  }

  /**
   * Retrieves the topK records from the vector store for the given query text and metadata filter.
   *
   * @param queryText the query text to retrieve records for.
   * @param topK the number of records to retrieve.
   * @param metadataFilter key-value pairs that candidate records must match.
   * @return the topK records from the vector store.
   */
  public List<RetrievalResult> retrieve(
      String queryText, int topK, Map<String, String> metadataFilter) {
    return retrieve(ImmutableList.of(new TextPart(queryText)), topK, metadataFilter);
  }

  /**
   * Retrieves the topK records from the vector store for the given query parts.
   *
   * @param queryParts the query parts to retrieve records for.
   * @param topK the number of records to retrieve.
   * @return the topK records from the vector store for the given query parts.
   */
  public List<RetrievalResult> retrieve(List<Part> queryParts, int topK) {
    return retrieve(queryParts, topK, /* metadataFilter= */ null);
  }

  /**
   * Retrieves the topK records from the vector store for the given query parts and metadata filter.
   *
   * @param queryParts the query parts to retrieve records for.
   * @param topK the number of records to retrieve.
   * @param metadataFilter key-value pairs that candidate records must match.
   * @return the topK records from the vector store.
   */
  public List<RetrievalResult> retrieve(
      List<Part> queryParts, int topK, Map<String, String> metadataFilter) {
    long timestamp = syntheticTimestamp.getAndIncrement();
    statsLogger.recordCpuInputArrival(timestamp);
    float[] queryEmbedding = embedContent(queryParts);
    List<RetrievalRecord> searchResults =
        components.vectorStore().search(queryEmbedding, topK, metadataFilter);

    List<RetrievalRecord> results = new ArrayList<>();
    Set<String> uniqueParentIds = new LinkedHashSet<>();
    for (RetrievalRecord record : searchResults) {
      String parentId = record.getParentId();
      if (parentId != null && !parentId.isEmpty()) {
        uniqueParentIds.add(parentId);
      }
    }

    Map<String, RetrievalRecord> parentRecordsMap = new HashMap<>();
    if (!uniqueParentIds.isEmpty()) {
      List<RetrievalRecord> parentRecords =
          components.vectorStore().get(ImmutableList.copyOf(uniqueParentIds));
      for (RetrievalRecord parent : parentRecords) {
        parentRecordsMap.put(parent.getId(), parent);
      }
    }

    Set<String> seenParents = new HashSet<>();
    for (RetrievalRecord record : searchResults) {
      String parentId = record.getParentId();
      if (parentId != null && !parentId.isEmpty()) {
        if (seenParents.add(parentId)) {
          RetrievalRecord parent = parentRecordsMap.get(parentId);
          if (parent != null) {
            // Populate the parent record with the highest-scoring child record's embedding.
            RetrievalRecord parentWithChildEmbedding =
                new RetrievalRecord(
                    parent.getId(),
                    parent.getContent(),
                    record.getEmbeddings(),
                    parent.getMetadata(),
                    parent.getParentId(),
                    parent.getChildIds());
            results.add(parentWithChildEmbedding);
          }
        }
      } else {
        if (seenParents.add(record.getId())) {
          results.add(record);
        }
      }
    }

    List<RetrievalResult> clientResults = new ArrayList<>();
    for (RetrievalRecord record : results) {
      Embedding query = Embedding.create(queryEmbedding, new byte[0], 0, Optional.empty());
      Embedding stored = Embedding.create(record.getEmbeddings(), new byte[0], 0, Optional.empty());
      double score = CosineSimilarity.compute(query, stored);
      clientResults.add(
          new RetrievalResult(record.getId(), record.getContent(), record.getMetadata(), score));
    }
    statsLogger.recordInvocationEnd(timestamp);
    return clientResults;
  }

  /**
   * Deletes the records from the vector store for the given ids.
   *
   * @param ids the ids of the records to delete.
   */
  public void delete(List<String> ids) {
    components.vectorStore().delete(ids);
  }

  /**
   * Deletes the records from the vector store matching the metadata filter.
   *
   * @param metadataFilter key-value pairs that records to delete must match.
   */
  public void delete(Map<String, String> metadataFilter) {
    components.vectorStore().delete(metadataFilter);
  }

  /** Deletes all records from the vector store. */
  public void deleteAll() {
    components.vectorStore().deleteAll();
  }

  /**
   * Retrieves all unique record identifiers in the vector store.
   *
   * @return a list of all unique record identifiers.
   */
  public List<String> getAllRecordIds() {
    return components.vectorStore().getAllRecordIds();
  }

  /** Closes the vector store. */
  @Override
  public void close() {
    statsLogger.logSessionEnd();
    if (components.vectorStore() != null) {
      try {
        components.vectorStore().close();
      } catch (Exception e) {
        // Ignore closing errors
      }
    }
  }
}
