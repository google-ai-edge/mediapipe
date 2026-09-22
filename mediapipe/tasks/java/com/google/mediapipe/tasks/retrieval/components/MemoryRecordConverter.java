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

import android.net.Uri;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.mediapipe.tasks.core.AudioPart;
import com.google.mediapipe.tasks.core.ImagePart;
import com.google.mediapipe.tasks.core.Part;
import com.google.mediapipe.tasks.core.TextPart;
import com.google.mediapipe.tasks.retrieval.model.RetrievalRecord;
import com.google.mediapipe.tasks.retrieval.semanticretriever.proto.MemoryProto;
import com.google.mediapipe.tasks.retrieval.semanticretriever.proto.MemoryProto.KeyValuePair;
import com.google.mediapipe.tasks.retrieval.semanticretriever.proto.MemoryProto.MemoryRecord;
import com.google.mediapipe.tasks.retrieval.semanticretriever.proto.MemoryProto.Metadata;
import com.google.mediapipe.tasks.retrieval.semanticretriever.proto.MemoryProto.Part.Kind;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Utility class for converting between RetrievalRecord and MemoryRecord protobuf. */
@SuppressWarnings({"IfChainToSwitch", "PatternMatchingInstanceof"})
public final class MemoryRecordConverter {
  private MemoryRecordConverter() {}

  /** Converts a RetrievalRecord to a MemoryRecord protobuf builder. */
  public static MemoryRecord toMemoryRecord(RetrievalRecord record) {
    Preconditions.checkArgument(!record.getContent().isEmpty(), "Record content cannot be empty.");
    MemoryRecord.Builder builder = MemoryRecord.newBuilder().setRecordId(record.getId());
    if (record.getParentId() != null && !record.getParentId().isEmpty()) {
      builder.setParentId(record.getParentId());
    }
    builder.addAllChildIds(record.getChildIds());

    for (float val : record.getEmbeddings()) {
      builder.addEmbeddings(val);
    }

    Metadata.Builder metadataBuilder = Metadata.newBuilder();
    for (Map.Entry<String, String> entry : record.getMetadata().entrySet()) {
      metadataBuilder.addKeyValuePairs(
          KeyValuePair.newBuilder().setKey(entry.getKey()).setValue(entry.getValue()).build());
    }
    builder.setMetadata(metadataBuilder.build());

    for (Part part : record.getContent()) {
      builder.addParts(toProtoPart(part));
    }
    if (builder.getPartsCount() == 1) {
      MemoryProto.Part part = builder.getParts(0);
      builder.setContentType(toContentType(part.getKind())).setText(part.getText());
    }

    return builder.build();
  }

  private static MemoryProto.Part toProtoPart(Part part) {
    MemoryProto.Part.Builder partBuilder = MemoryProto.Part.newBuilder();
    if (part instanceof TextPart) {
      TextPart textPart = (TextPart) part;
      partBuilder.setKind(Kind.TEXT).setText(Strings.nullToEmpty(textPart.getText()));
    } else if (part instanceof ImagePart) {
      ImagePart imagePart = (ImagePart) part;
      partBuilder
          .setKind(Kind.IMAGE)
          .setText(imagePart.filePath() != null ? imagePart.filePath().toString() : "");
    } else if (part instanceof AudioPart) {
      AudioPart audioPart = (AudioPart) part;
      partBuilder
          .setKind(Kind.AUDIO)
          .setText(audioPart.filePath() != null ? audioPart.filePath().toString() : "");
    }
    return partBuilder.build();
  }

  private static String toContentType(Kind kind) {
    switch (kind) {
      case IMAGE:
        return VectorStoreRecord.ContentType.IMAGE.name();
      case AUDIO:
        return VectorStoreRecord.ContentType.AUDIO.name();
      case TEXT:
      default:
        return VectorStoreRecord.ContentType.TEXT.name();
    }
  }

  /** Converts a RetrievalRecord to serialized MemoryRecord protobuf bytes. */
  public static byte[] toMemoryRecordProtoBytes(RetrievalRecord record) {
    return toMemoryRecord(record).toByteArray();
  }

  /** Converts a MemoryRecord protobuf instance to a RetrievalRecord. */
  public static RetrievalRecord toRetrievalRecord(MemoryRecord memoryRecord) {
    String recordId = memoryRecord.getRecordId();
    String parentId = memoryRecord.hasParentId() ? memoryRecord.getParentId() : null;
    List<String> childIds = memoryRecord.getChildIdsList();

    Map<String, String> userMetadata = new HashMap<>();
    if (memoryRecord.hasMetadata()) {
      for (KeyValuePair pair : memoryRecord.getMetadata().getKeyValuePairsList()) {
        userMetadata.put(pair.getKey(), pair.getValue());
      }
    }

    float[] embeddings = new float[memoryRecord.getEmbeddingsList().size()];
    for (int i = 0; i < embeddings.length; i++) {
      embeddings[i] = memoryRecord.getEmbeddings(i);
    }

    List<Part> content = new ArrayList<>();
    for (MemoryProto.Part protoPart : memoryRecord.getPartsList()) {
      switch (protoPart.getKind()) {
        case IMAGE:
          Uri imgUri =
              protoPart.hasText() && !protoPart.getText().isEmpty()
                  ? Uri.parse(protoPart.getText())
                  : null;
          content.add(new ImagePart(imgUri));
          break;
        case AUDIO:
          Uri audioUri =
              protoPart.hasText() && !protoPart.getText().isEmpty()
                  ? Uri.parse(protoPart.getText())
                  : null;
          content.add(new AudioPart(audioUri));
          break;
        case TEXT:
        default:
          content.add(new TextPart(protoPart.hasText() ? protoPart.getText() : ""));
          break;
      }
    }

    return new RetrievalRecord(recordId, content, embeddings, userMetadata, parentId, childIds);
  }

  /** Parses serialized MemoryRecord protobuf bytes into a RetrievalRecord. */
  public static RetrievalRecord toRetrievalRecord(byte[] storeRecordBytes) {
    try {
      MemoryRecord memoryRecord =
          MemoryRecord.parseFrom(storeRecordBytes, ExtensionRegistryLite.getEmptyRegistry());
      return toRetrievalRecord(memoryRecord);
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException("Failed to parse MemoryRecord protobuf bytes", e);
    }
  }
}
