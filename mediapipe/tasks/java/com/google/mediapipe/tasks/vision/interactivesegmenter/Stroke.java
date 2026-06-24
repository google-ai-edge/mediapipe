// Copyright 2026 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.vision.interactivesegmenter;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint;
import java.util.List;

/** Represents a single user-drawn stroke for interactive segmentation. */
@AutoValue
public abstract class Stroke {

  /** Specifies the polarity or mode of a brush stroke. */
  public enum BrushMode {
    /** Unspecified brush mode. */
    UNSPECIFIED(0),
    /** Positive brush mode, meaning the stroke adds to the selection. */
    POSITIVE(1),
    /** Negative brush mode, meaning the stroke subtracts from the selection. */
    NEGATIVE(2),
    /** Lasso brush mode. */
    LASSO(3);

    private final int value;

    BrushMode(int value) {
      this.value = value;
    }

    public int getValue() {
      return value;
    }
  }

  /** The brush mode used for this stroke. */
  public abstract BrushMode brushMode();

  /** The sequence of normalized key points forming the stroke. */
  public abstract ImmutableList<NormalizedKeypoint> points();

  /** Whether the stroke is complete. */
  public abstract boolean completed();

  /** Creates a builder for {@link Stroke}. */
  public static Builder builder() {
    return new AutoValue_Stroke.Builder();
  }

  /** Builder for {@link Stroke}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Sets the brush mode used for this stroke. */
    public abstract Builder setBrushMode(BrushMode brushMode);

    /** Sets the sequence of normalized key points forming the stroke. */
    public abstract Builder setPoints(List<NormalizedKeypoint> points);

    /** Sets whether the stroke is complete. */
    public abstract Builder setCompleted(boolean completed);

    /** Builds the {@link Stroke} instance. */
    public abstract Stroke build();
  }
}
