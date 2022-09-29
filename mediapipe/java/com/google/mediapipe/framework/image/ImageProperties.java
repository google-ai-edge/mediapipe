/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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
import com.google.mediapipe.framework.image.Image.ImageFormat;
import com.google.mediapipe.framework.image.Image.StorageType;

/** Groups a set of properties to describe how an image is stored. */
@AutoValue
public abstract class ImageProperties {

  /**
   * Gets the pixel format of the image.
   *
   * @see Image.ImageFormat
   */
  @ImageFormat
  public abstract int getImageFormat();

  /**
   * Gets the storage type of the image.
   *
   * @see Image.StorageType
   */
  @StorageType
  public abstract int getStorageType();

  @Memoized
  @Override
  public abstract int hashCode();

  /**
   * Creates a builder of {@link ImageProperties}.
   *
   * @see ImageProperties.Builder
   */
  static Builder builder() {
    return new AutoValue_ImageProperties.Builder();
  }

  /** Builds a {@link ImageProperties}. */
  @AutoValue.Builder
  abstract static class Builder {

    /**
     * Sets the {@link Image.ImageFormat}.
     *
     * @see ImageProperties#getImageFormat
     */
    abstract Builder setImageFormat(@ImageFormat int value);

    /**
     * Sets the {@link Image.StorageType}.
     *
     * @see ImageProperties#getStorageType
     */
    abstract Builder setStorageType(@StorageType int value);

    /** Builds the {@link ImageProperties}. */
    abstract ImageProperties build();
  }

  // Hide the constructor.
  ImageProperties() {}
}
