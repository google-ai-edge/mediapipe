// Copyright 2022 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.tasks.components.containers;

import com.google.auto.value.AutoValue;
import com.google.mediapipe.tasks.components.containers.proto.EmbeddingsProto;
import java.util.Optional;

/**
 * Represents the embedding for a given embedder head. Typically used in embedding tasks.
 *
 * <p>One and only one of the two 'floatEmbedding' and 'quantizedEmbedding' will contain data, based
 * on whether or not the embedder was configured to perform scala quantization.
 */
@AutoValue
public abstract class Embedding {

  /**
   * Creates an {@link Embedding} instance.
   *
   * @param floatEmbedding the floating-point embedding
   * @param quantizedEmbedding the quantized embedding.
   * @param headIndex the index of the embedder head.
   * @param headName the optional name of the embedder head.
   */
  public static Embedding create(
      float[] floatEmbedding, byte[] quantizedEmbedding, int headIndex, Optional<String> headName) {
    return new AutoValue_Embedding(floatEmbedding, quantizedEmbedding, headIndex, headName);
  }

  /**
   * Creates an {@link Embedding} object from an {@link EmbeddingsProto.Embedding} protobuf message.
   *
   * @param proto the {@link EmbeddingsProto.Embedding} protobuf message to convert.
   */
  public static Embedding createFromProto(EmbeddingsProto.Embedding proto) {
    float[] floatEmbedding;
    if (proto.hasFloatEmbedding()) {
      floatEmbedding = new float[proto.getFloatEmbedding().getValuesCount()];
      for (int i = 0; i < floatEmbedding.length; i++) {
        floatEmbedding[i] = proto.getFloatEmbedding().getValues(i);
      }
    } else {
      floatEmbedding = new float[0];
    }
    return Embedding.create(
        floatEmbedding,
        proto.hasQuantizedEmbedding()
            ? proto.getQuantizedEmbedding().getValues().toByteArray()
            : new byte[0],
        proto.getHeadIndex(),
        proto.hasHeadName() ? Optional.of(proto.getHeadName()) : Optional.empty());
  }

  /**
   * Floating-point embedding.
   *
   * <p>Empty if the embedder was configured to perform scalar quantization.
   */
  public abstract float[] floatEmbedding();

  /**
   * Quantized embedding.
   *
   * <p>Empty if the embedder was not configured to perform scalar quantization.
   */
  public abstract byte[] quantizedEmbedding();

  /**
   * The index of the embedder head these entries refer to. This is useful for multi-head models.
   */
  public abstract int headIndex();

  /** The optional name of the embedder head, which is the corresponding tensor metadata name. */
  public abstract Optional<String> headName();
}
