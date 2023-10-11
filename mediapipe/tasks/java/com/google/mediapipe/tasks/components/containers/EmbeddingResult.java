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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/** Represents the embedding results of a model. Typically used as a result for embedding tasks. */
@AutoValue
public abstract class EmbeddingResult {

  /**
   * Creates a {@link EmbeddingResult} instance.
   *
   * @param embeddings the list of {@link Embedding} objects containing the embedding for each head
   *     of the model.
   * @param timestampMs the optional timestamp (in milliseconds) of the start of the chunk of data
   *     corresponding to these results.
   */
  public static EmbeddingResult create(List<Embedding> embeddings, Optional<Long> timestampMs) {
    return new AutoValue_EmbeddingResult(Collections.unmodifiableList(embeddings), timestampMs);
  }

  /**
   * Creates a {@link EmbeddingResult} object from a {@link EmbeddingsProto.EmbeddingResult}
   * protobuf message.
   *
   * @param proto the {@link EmbeddingsProto.EmbeddingResult} protobuf message to convert.
   */
  public static EmbeddingResult createFromProto(EmbeddingsProto.EmbeddingResult proto) {
    List<Embedding> embeddings = new ArrayList<>();
    for (EmbeddingsProto.Embedding embeddingProto : proto.getEmbeddingsList()) {
      embeddings.add(Embedding.createFromProto(embeddingProto));
    }
    Optional<Long> timestampMs =
        proto.hasTimestampMs() ? Optional.of(proto.getTimestampMs()) : Optional.empty();
    return create(embeddings, timestampMs);
  }

  /** The embedding results for each head of the model. */
  public abstract List<Embedding> embeddings();

  /**
   * The optional timestamp (in milliseconds) of the start of the chunk of data corresponding to
   * these results.
   *
   * <p>This is only used for embedding extraction on time series (e.g. audio embedder). In these
   * use cases, the amount of data to process might exceed the maximum size that the model can
   * process: to solve this, the input data is split into multiple chunks starting at different
   * timestamps.
   */
  public abstract Optional<Long> timestampMs();
}
