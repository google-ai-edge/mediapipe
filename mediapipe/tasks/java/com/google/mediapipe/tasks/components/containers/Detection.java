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

import android.graphics.RectF;
import com.google.auto.value.AutoValue;
import java.util.Collections;
import java.util.List;

/**
 * Represents one detected object in the results of {@link
 * com.google.mediapipe.tasks.version.objectdetector.ObjectDetector}.
 */
@AutoValue
public abstract class Detection {

  /**
   * Creates a {@link Detection} instance from a list of {@link Category} and a bounding box.
   *
   * @param categories a list of {@link Category} objects that contain category name, display name,
   *     score, and the label index.
   * @param boundingBox a {@link RectF} object to represent the bounding box.
   */
  public static Detection create(List<Category> categories, RectF boundingBox) {

    // As an open source project, we've been trying avoiding depending on common java libraries,
    // such as Guava, because it may introduce conflicts with clients who also happen to use those
    // libraries. Therefore, instead of using ImmutableList here, we convert the List into
    // unmodifiableList
    return new AutoValue_Detection(Collections.unmodifiableList(categories), boundingBox);
  }

  /** A list of {@link Category} objects. */
  public abstract List<Category> categories();

  /** A {@link RectF} object to represent the bounding box of the detected object. */
  public abstract RectF boundingBox();
}
