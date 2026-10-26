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

package com.google.mediapipe.tasks.vision.overlay;

import android.graphics.Color;

/**
 * Colors and sizes for {@link VisionOverlayRenderer}. Values are in pixels; scale them by {@code
 * density} in Compose ({@code LocalDensity.current.density}) or from {@code
 * DisplayMetrics.density}.
 */
public final class OverlayStyle {
  /** MediaPipe teal used for connections and boxes. */
  public static final int DEFAULT_CONNECTION_COLOR = 0xFF00A3A1;

  /** Landmark dots. */
  public static final int DEFAULT_LANDMARK_COLOR = Color.YELLOW;

  /** Detection box stroke. */
  public static final int DEFAULT_BOX_COLOR = 0xFF0077CC;

  private final int landmarkColor;
  private final int connectionColor;
  private final int boxColor;
  private final int textColor;
  private final int textBackgroundColor;
  private final float landmarkRadius;
  private final float connectionStrokeWidth;
  private final float boxStrokeWidth;
  private final float textSize;
  private final float textPadding;

  OverlayStyle(Builder builder) {
    this.landmarkColor = builder.landmarkColor;
    this.connectionColor = builder.connectionColor;
    this.boxColor = builder.boxColor;
    this.textColor = builder.textColor;
    this.textBackgroundColor = builder.textBackgroundColor;
    this.landmarkRadius = builder.landmarkRadius;
    this.connectionStrokeWidth = builder.connectionStrokeWidth;
    this.boxStrokeWidth = builder.boxStrokeWidth;
    this.textSize = builder.textSize;
    this.textPadding = builder.textPadding;
  }

  /** Density-independent defaults (1px = 1dp). Prefer {@link #mediapipeDefault(float)}. */
  public static OverlayStyle mediapipeDefault() {
    return mediapipeDefault(/* density= */ 1.f);
  }

  /**
   * Defaults scaled by display density so strokes stay readable on Compose / high-dpi screens.
   *
   * @param density {@code Resources.getDisplayMetrics().density} or Compose {@code
   *     LocalDensity.current.density}
   */
  public static OverlayStyle mediapipeDefault(float density) {
    if (density <= 0.f) {
      throw new IllegalArgumentException("Density must be positive, found: " + density);
    }
    return builder()
        .setLandmarkRadius(4.f * density)
        .setConnectionStrokeWidth(4.f * density)
        .setBoxStrokeWidth(4.f * density)
        .setTextSize(16.f * density)
        .setTextPadding(4.f * density)
        .build();
  }

  public static Builder builder() {
    return new Builder();
  }

  public int landmarkColor() {
    return landmarkColor;
  }

  public int connectionColor() {
    return connectionColor;
  }

  public int boxColor() {
    return boxColor;
  }

  public int textColor() {
    return textColor;
  }

  public int textBackgroundColor() {
    return textBackgroundColor;
  }

  public float landmarkRadius() {
    return landmarkRadius;
  }

  public float connectionStrokeWidth() {
    return connectionStrokeWidth;
  }

  public float boxStrokeWidth() {
    return boxStrokeWidth;
  }

  public float textSize() {
    return textSize;
  }

  public float textPadding() {
    return textPadding;
  }

  public Builder toBuilder() {
    return builder()
        .setLandmarkColor(landmarkColor)
        .setConnectionColor(connectionColor)
        .setBoxColor(boxColor)
        .setTextColor(textColor)
        .setTextBackgroundColor(textBackgroundColor)
        .setLandmarkRadius(landmarkRadius)
        .setConnectionStrokeWidth(connectionStrokeWidth)
        .setBoxStrokeWidth(boxStrokeWidth)
        .setTextSize(textSize)
        .setTextPadding(textPadding);
  }

  /** Builder for {@link OverlayStyle}. */
  public static final class Builder {
    private int landmarkColor = DEFAULT_LANDMARK_COLOR;
    private int connectionColor = DEFAULT_CONNECTION_COLOR;
    private int boxColor = DEFAULT_BOX_COLOR;
    private int textColor = Color.WHITE;
    private int textBackgroundColor = 0x99000000;
    private float landmarkRadius = 4.f;
    private float connectionStrokeWidth = 4.f;
    private float boxStrokeWidth = 4.f;
    private float textSize = 16.f;
    private float textPadding = 4.f;

    public Builder setLandmarkColor(int landmarkColor) {
      this.landmarkColor = landmarkColor;
      return this;
    }

    public Builder setConnectionColor(int connectionColor) {
      this.connectionColor = connectionColor;
      return this;
    }

    public Builder setBoxColor(int boxColor) {
      this.boxColor = boxColor;
      return this;
    }

    public Builder setTextColor(int textColor) {
      this.textColor = textColor;
      return this;
    }

    public Builder setTextBackgroundColor(int textBackgroundColor) {
      this.textBackgroundColor = textBackgroundColor;
      return this;
    }

    public Builder setLandmarkRadius(float landmarkRadius) {
      this.landmarkRadius = requirePositive("landmarkRadius", landmarkRadius);
      return this;
    }

    public Builder setConnectionStrokeWidth(float connectionStrokeWidth) {
      this.connectionStrokeWidth = requirePositive("connectionStrokeWidth", connectionStrokeWidth);
      return this;
    }

    public Builder setBoxStrokeWidth(float boxStrokeWidth) {
      this.boxStrokeWidth = requirePositive("boxStrokeWidth", boxStrokeWidth);
      return this;
    }

    public Builder setTextSize(float textSize) {
      this.textSize = requirePositive("textSize", textSize);
      return this;
    }

    public Builder setTextPadding(float textPadding) {
      if (textPadding < 0.f) {
        throw new IllegalArgumentException("textPadding must be >= 0, found: " + textPadding);
      }
      this.textPadding = textPadding;
      return this;
    }

    public OverlayStyle build() {
      return new OverlayStyle(this);
    }

    private static float requirePositive(String name, float value) {
      if (value <= 0.f) {
        throw new IllegalArgumentException(name + " must be positive, found: " + value);
      }
      return value;
    }
  }
}
