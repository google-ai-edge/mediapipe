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
import com.google.mediapipe.tasks.components.containers.proto.ClassificationsProto;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Represents the classification results of a model. Typically used as a result for classification
 * tasks.
 */
@AutoValue
public abstract class ClassificationResult {

  /**
   * Creates a {@link ClassificationResult} instance.
   *
   * @param classifications the list of {@link Classifications} objects containing the predicted
   *     categories for each head of the model.
   * @param timestampMs the optional timestamp (in milliseconds) of the start of the chunk of data
   *     corresponding to these results.
   */
  public static ClassificationResult create(
      List<Classifications> classifications, Optional<Long> timestampMs) {
    return new AutoValue_ClassificationResult(
        Collections.unmodifiableList(classifications), timestampMs);
  }

  /**
   * Creates a {@link ClassificationResult} object from a {@link
   * ClassificationsProto.ClassificationResult} protobuf message.
   *
   * @param proto the {@link ClassificationsProto.ClassificationResult} protobuf message to convert.
   */
  public static ClassificationResult createFromProto(
      ClassificationsProto.ClassificationResult proto) {
    List<Classifications> classifications = new ArrayList<>();
    for (ClassificationsProto.Classifications classificationsProto :
        proto.getClassificationsList()) {
      classifications.add(Classifications.createFromProto(classificationsProto));
    }
    Optional<Long> timestampMs =
        proto.hasTimestampMs() ? Optional.of(proto.getTimestampMs()) : Optional.empty();
    return create(classifications, timestampMs);
  }

  /** The classification results for each head of the model. */
  public abstract List<Classifications> classifications();

  /**
   * The optional timestamp (in milliseconds) of the start of the chunk of data corresponding to
   * these results.
   *
   * <p>This is only used for classification on time series (e.g. audio classification). In these
   * use cases, the amount of data to process might exceed the maximum size that the model can
   * process: to solve this, the input data is split into multiple chunks starting at different
   * timestamps.
   */
  public abstract Optional<Long> timestampMs();
}
