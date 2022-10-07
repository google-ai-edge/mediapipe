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
 * Represents a list of predicted categories with an optional timestamp. Typically used as result
 * for classification tasks.
 */
@AutoValue
public abstract class ClassificationEntry {
  /**
   * Creates a {@link ClassificationEntry} instance from a list of {@link Category} and optional
   * timestamp.
   *
   * @param categories the list of {@link Category} objects that contain category name, display
   *     name, score and label index.
   * @param timestampMs the {@link long} representing the timestamp for which these categories were
   *     obtained.
   */
  public static ClassificationEntry create(List<Category> categories, long timestampMs) {
    return new AutoValue_ClassificationEntry(Collections.unmodifiableList(categories), timestampMs);
  }

  /** The list of predicted {@link Category} objects, sorted by descending score. */
  public abstract List<Category> categories();

  /**
   * The timestamp (in milliseconds) associated to the classification entry. This is useful for time
   * series use cases, e.g. audio classification.
   */
  public abstract long timestampMs();
}
