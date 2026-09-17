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

package com.google.mediapipe.tasks.retrieval.model;

import android.net.Uri;
import com.google.mediapipe.tasks.core.Part;

/** Represents an image part of a multi-modal content block. */
public final class ImagePart extends Part {
  private final Uri filePath;

  /**
   * Creates a new {@link ImagePart} with the specified image.
   *
   * @param filePath the image file path.
   */
  public ImagePart(Uri filePath) {
    this.filePath = filePath;
  }

  /**
   * Returns the file path of this image.
   *
   * @return the file path.
   */
  public Uri filePath() {
    return filePath;
  }
}
