/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

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

package com.google.mediapipe.tasks.core;

import android.net.Uri;
import androidx.annotation.Nullable;

/** Represents an image part of a multi-modal content block. */
public final class ImagePart extends Part {
  @Nullable private final Uri filePath;
  @Nullable private final byte[] imageBytes;

  /**
   * Creates a new {@link ImagePart} with the specified image.
   *
   * @param filePath the image file path.
   */
  public ImagePart(Uri filePath) {
    this(filePath, null);
  }

  /**
   * Creates a new {@link ImagePart} with the specified in-memory encoded image bytes.
   *
   * @param imageBytes the encoded image bytes (for example, PNG or JPEG).
   */
  public ImagePart(byte[] imageBytes) {
    this(null, imageBytes);
  }

  /**
   * Creates a new {@link ImagePart} with the specified image file path and optional in-memory
   * encoded image bytes.
   *
   * @param filePath the optional image file path.
   * @param imageBytes the optional encoded image bytes.
   */
  public ImagePart(@Nullable Uri filePath, @Nullable byte[] imageBytes) {
    this.filePath = filePath;
    this.imageBytes = imageBytes != null ? imageBytes.clone() : null;
  }

  /**
   * Returns the file path of this image, or {@code null} if stored in-memory.
   *
   * @return the file path.
   */
  @Nullable
  public Uri getFilePath() {
    return filePath;
  }

  /**
   * Returns the file path of this image, or {@code null} if stored in-memory.
   *
   * @return the file path.
   */
  @Nullable
  public Uri filePath() {
    return filePath;
  }

  /**
   * Returns a defensive copy of the in-memory encoded image bytes, or {@code null} if not present.
   *
   * @return the encoded image bytes.
   */
  @Nullable
  public byte[] imageBytes() {
    return imageBytes != null ? imageBytes.clone() : null;
  }
}
