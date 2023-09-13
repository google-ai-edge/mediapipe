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

package com.google.mediapipe.tasks.vision.core;

import android.graphics.RectF;
import com.google.auto.value.AutoValue;
import java.util.Optional;

// TODO: add support for image flipping.
/** Options for image processing. */
@AutoValue
public abstract class ImageProcessingOptions {

  /**
   * Builder for {@link ImageProcessingOptions}.
   *
   * <p>If both region-of-interest and rotation are specified, the crop around the
   * region-of-interest is extracted first, then the specified rotation is applied to the crop.
   */
  @AutoValue.Builder
  public abstract static class Builder {
    /**
     * Sets the optional region-of-interest to crop from the image. If not specified, the full image
     * is used.
     *
     * <p>Coordinates must be in [0,1], {@code left} must be < {@code right} and {@code top} must be
     * < {@code bottom}, otherwise an IllegalArgumentException will be thrown when {@link #build()}
     * is called.
     */
    public abstract Builder setRegionOfInterest(RectF value);

    /**
     * Sets the rotation to apply to the image (or cropped region-of-interest), in degrees
     * clockwise. Defaults to 0.
     *
     * <p>The rotation must be a multiple (positive or negative) of 90°, otherwise an
     * IllegalArgumentException will be thrown when {@link #build()} is called.
     */
    public abstract Builder setRotationDegrees(int value);

    abstract ImageProcessingOptions autoBuild();

    /**
     * Validates and builds the {@link ImageProcessingOptions} instance.
     *
     * @throws IllegalArgumentException if some of the provided values do not meet their
     *     requirements.
     */
    public final ImageProcessingOptions build() {
      ImageProcessingOptions options = autoBuild();
      if (options.regionOfInterest().isPresent()) {
        RectF roi = options.regionOfInterest().get();
        if (roi.left >= roi.right || roi.top >= roi.bottom) {
          throw new IllegalArgumentException(
              String.format(
                  "Expected left < right and top < bottom, found: %s.", roi.toShortString()));
        }
        if (roi.left < 0 || roi.right > 1 || roi.top < 0 || roi.bottom > 1) {
          throw new IllegalArgumentException(
              String.format("Expected RectF values in [0,1], found: %s.", roi.toShortString()));
        }
      }
      if (options.rotationDegrees() % 90 != 0) {
        throw new IllegalArgumentException(
            String.format(
                "Expected rotation to be a multiple of 90°, found: %d.",
                options.rotationDegrees()));
      }
      return options;
    }
  }

  public abstract Optional<RectF> regionOfInterest();

  public abstract int rotationDegrees();

  public static Builder builder() {
    return new AutoValue_ImageProcessingOptions.Builder().setRotationDegrees(0);
  }
}
