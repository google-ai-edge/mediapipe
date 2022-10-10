// Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
import java.util.Collections;
import java.util.List;

/**
 * Represents the list of classification for a given classifier head. Typically used as a result for
 * classification tasks.
 */
@AutoValue
public abstract class Classifications {

  /**
   * Creates a {@link Classifications} instance.
   *
   * @param entries the list of {@link ClassificationEntry} objects containing the predicted
   *     categories.
   * @param headIndex the index of the classifier head.
   * @param headName the name of the classifier head.
   */
  public static Classifications create(
      List<ClassificationEntry> entries, int headIndex, String headName) {
    return new AutoValue_Classifications(
        Collections.unmodifiableList(entries), headIndex, headName);
  }

  /** A list of {@link ClassificationEntry} objects. */
  public abstract List<ClassificationEntry> entries();

  /**
   * The index of the classifier head these entries refer to. This is useful for multi-head models.
   */
  public abstract int headIndex();

  /** The name of the classifier head, which is the corresponding tensor metadata name. */
  public abstract String headName();
}
