/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.mediapipe.framework.image;

import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.mediapipe.framework.image.MPImage.MPImageFormat;
import com.google.mediapipe.framework.image.MPImage.StorageType;

/** Groups a set of properties to describe how an image is stored. */
@AutoValue
public abstract class MPImageProperties {

  /**
   * Gets the pixel format of the image.
   *
   * @see MPImage.MPImageFormat
   */
  @MPImageFormat
  public abstract int getImageFormat();

  /**
   * Gets the storage type of the image.
   *
   * @see MPImage.StorageType
   */
  @StorageType
  public abstract int getStorageType();

  @Memoized
  @Override
  public abstract int hashCode();

  /**
   * Creates a builder of {@link MPImageProperties}.
   *
   * @see MPImageProperties.Builder
   */
  static Builder builder() {
    return new AutoValue_MPImageProperties.Builder();
  }

  /** Builds a {@link MPImageProperties}. */
  @AutoValue.Builder
  abstract static class Builder {

    /**
     * Sets the {@link MPImage.MPImageFormat}.
     *
     * @see MPImageProperties#getImageFormat
     */
    abstract Builder setImageFormat(@MPImageFormat int value);

    /**
     * Sets the {@link MPImage.StorageType}.
     *
     * @see MPImageProperties#getStorageType
     */
    abstract Builder setStorageType(@StorageType int value);

    /** Builds the {@link MPImageProperties}. */
    abstract MPImageProperties build();
  }

  // Hide the constructor.
  MPImageProperties() {}
}
