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

import android.graphics.Bitmap;

/**
 * Utility for extracting {@link android.graphics.Bitmap} from {@link MPImage}.
 *
 * <p>Currently it only supports {@link MPImage} with {@link MPImage#STORAGE_TYPE_BITMAP}, otherwise
 * {@link IllegalArgumentException} will be thrown.
 */
public final class BitmapExtractor {

  /**
   * Extracts a {@link android.graphics.Bitmap} from a {@link MPImage}.
   *
   * @param image the image to extract {@link android.graphics.Bitmap} from.
   * @return the {@link android.graphics.Bitmap} stored in {@link MPImage}
   * @throws IllegalArgumentException when the extraction requires unsupported format or data type
   *     conversions.
   */
  public static Bitmap extract(MPImage image) {
    MPImageContainer imageContainer = image.getContainer(MPImage.STORAGE_TYPE_BITMAP);
    if (imageContainer != null) {
      return ((BitmapImageContainer) imageContainer).getBitmap();
    } else {
      // TODO: Support ByteBuffer -> Bitmap conversion.
      throw new IllegalArgumentException(
          "Extracting Bitmap from a MPImage created by objects other than Bitmap is not"
              + " supported");
    }
  }

  private BitmapExtractor() {}
}
