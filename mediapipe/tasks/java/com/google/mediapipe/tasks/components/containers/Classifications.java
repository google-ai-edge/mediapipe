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
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Represents the list of classification for a given classifier head. Typically used as a result for
 * classification tasks.
 */
@AutoValue
public abstract class Classifications {

  /**
   * Creates a {@link Classifications} instance.
   *
   * @param categories the list of {@link Category} objects containing the predicted categories.
   * @param headIndex the index of the classifier head.
   * @param headName the optional name of the classifier head.
   */
  public static Classifications create(
      List<Category> categories, int headIndex, Optional<String> headName) {
    return new AutoValue_Classifications(
        Collections.unmodifiableList(categories), headIndex, headName);
  }

  /**
   * Creates a {@link Classifications} object from a {@link ClassificationsProto.Classifications}
   * protobuf message.
   *
   * @param proto the {@link ClassificationsProto.Classifications} protobuf message to convert.
   */
  public static Classifications createFromProto(ClassificationsProto.Classifications proto) {
    List<Category> categories = Category.createListFromProto(proto.getClassificationList());
    Optional<String> headName =
        proto.hasHeadName() ? Optional.of(proto.getHeadName()) : Optional.empty();
    return create(categories, proto.getHeadIndex(), headName);
  }

  /** A list of {@link Category} objects. */
  public abstract List<Category> categories();

  /**
   * The index of the classifier head these entries refer to. This is useful for multi-head models.
   */
  public abstract int headIndex();

  /** The optional name of the classifier head, which is the corresponding tensor metadata name. */
  public abstract Optional<String> headName();
}
