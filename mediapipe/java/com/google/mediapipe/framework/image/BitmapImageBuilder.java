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

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import java.io.IOException;

/**
 * Builds {@link MPImage} from {@link android.graphics.Bitmap}.
 *
 * <p>You can pass in either mutable or immutable {@link android.graphics.Bitmap}. However once
 * {@link android.graphics.Bitmap} is passed in, to keep data integrity you shouldn't modify content
 * in it.
 *
 * <p>Use {@link BitmapExtractor} to get {@link android.graphics.Bitmap} you passed in.
 */
public class BitmapImageBuilder {

  // Mandatory fields.
  private final Bitmap bitmap;

  // Optional fields.
  private long timestamp;

  /**
   * Creates the builder with a mandatory {@link android.graphics.Bitmap}.
   *
   * @param bitmap image data object.
   */
  public BitmapImageBuilder(Bitmap bitmap) {
    this.bitmap = bitmap;
    timestamp = 0;
  }

  /**
   * Creates the builder to build {@link MPImage} from a file.
   *
   * @param context the application context.
   * @param uri the path to the resource file.
   */
  public BitmapImageBuilder(Context context, Uri uri) throws IOException {
    this(MediaStore.Images.Media.getBitmap(context.getContentResolver(), uri));
  }

  /** Sets value for {@link MPImage#getTimestamp()}. */
  BitmapImageBuilder setTimestamp(long timestamp) {
    this.timestamp = timestamp;
    return this;
  }

  /** Builds a {@link MPImage} instance. */
  public MPImage build() {
    return new MPImage(
        new BitmapImageContainer(bitmap), timestamp, bitmap.getWidth(), bitmap.getHeight());
  }
}
