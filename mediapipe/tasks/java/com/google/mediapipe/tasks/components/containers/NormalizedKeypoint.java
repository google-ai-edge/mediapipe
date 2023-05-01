// Copyright 2023 The MediaPipe Authors.
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
import java.util.Objects;
import java.util.Optional;

/**
 * Normalized keypoint represents a point in 2D space with x, y coordinates. x and y are normalized
 * to [0.0, 1.0] by the image width and height respectively.
 */
@AutoValue
public abstract class NormalizedKeypoint {
  private static final float TOLERANCE = 1e-6f;

  /**
   * Creates a {@link NormalizedKeypoint} instance from normalized x and y coordinates.
   *
   * @param x the x coordinates of the normalized keypoint.
   * @param y the y coordinates of the normalized keypoint.
   */
  public static NormalizedKeypoint create(float x, float y) {
    return new AutoValue_NormalizedKeypoint(x, y, Optional.empty(), Optional.empty());
  }

  /**
   * Creates a {@link NormalizedKeypoint} instance from normalized x and y coordinates, and the
   * optional label and keypoint score.
   *
   * @param x the x coordinates of the normalized keypoint.
   * @param y the y coordinates of the normalized keypoint.
   * @param label optional label of the keypoint.
   * @param score optional score of the keypoint.
   */
  public static NormalizedKeypoint create(
      float x, float y, Optional<String> label, Optional<Float> score) {
    return new AutoValue_NormalizedKeypoint(x, y, label, score);
  }

  // The x coordinates of the normalized keypoint.
  public abstract float x();

  // The y coordinates of the normalized keypoint.
  public abstract float y();

  // optional label of the keypoint.
  public abstract Optional<String> label();

  // optional score of the keypoint.
  public abstract Optional<Float> score();

  @Override
  public final boolean equals(Object o) {
    if (!(o instanceof NormalizedKeypoint)) {
      return false;
    }
    NormalizedKeypoint other = (NormalizedKeypoint) o;
    return Math.abs(other.x() - this.x()) < TOLERANCE && Math.abs(other.x() - this.y()) < TOLERANCE;
  }

  @Override
  public final int hashCode() {
    return Objects.hash(x(), y(), label(), score());
  }

  @Override
  public final String toString() {
    return "<Normalized Keypoint (x="
        + x()
        + " y="
        + y()
        + " label= "
        + label().orElse("null")
        + " score="
        + (score().isPresent() ? score().get() : "null")
        + ")>";
  }
}
