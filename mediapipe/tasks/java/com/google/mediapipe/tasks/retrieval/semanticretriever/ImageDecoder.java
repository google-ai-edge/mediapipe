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

package com.google.mediapipe.tasks.retrieval.semanticretriever;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import androidx.annotation.Nullable;
import com.google.common.io.ByteStreams;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

/** Stateless utility class managing the decoding of images from a Uri. */
public final class ImageDecoder {

  private ImageDecoder() {}

  /**
   * Decodes an {@link MPImage} from the specified Uri.
   *
   * @param context the Android context.
   * @param uri the Uri of the image file.
   * @return the decoded MPImage, or null if decoding failed.
   */
  @Nullable
  @SuppressWarnings("EnumOrdinal")
  public static MPImage decodeImage(Context context, Uri uri) {
    byte[] bytes = getImageBytes(context, uri);
    if (bytes == null) {
      return null;
    }
    Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
    if (bitmap == null) {
      return null;
    }
    return new BitmapImageBuilder(bitmap).build();
  }

  /**
   * Reads all bytes of the image file from the specified Uri.
   *
   * @param context the Android context.
   * @param uri the Uri of the image file.
   * @return the raw bytes of the image, or null if reading failed.
   */
  @Nullable
  @SuppressWarnings("EnumOrdinal")
  public static byte[] getImageBytes(Context context, Uri uri) {
    if (uri == null) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INVALID_ARGUMENT.ordinal(), "URI cannot be null");
    }
    byte[] bytes = null;
    String scheme = uri.getScheme();
    try {
      if (scheme != null && (scheme.equals("content") || scheme.equals("android.resource"))) {
        if (context != null) {
          // mediapipe:oss-insert-begin(oss-only)
          // try (InputStream is = context.getContentResolver().openInputStream(uri)) {
          //   bytes = ByteStreams.toByteArray(is);
          // }
          // mediapipe:oss-insert-end
        }
      } else if (scheme == null || scheme.equals("file")) {
        if (context != null) {
          String path = uri.getPath();
          if (path != null && path.startsWith("/android_asset/")) {
            String assetPath = path.substring("/android_asset/".length());
            try (InputStream is = context.getAssets().open(assetPath)) {
              bytes = ByteStreams.toByteArray(is);
            }
          } else {
            Uri safeUri = scheme == null ? uri.buildUpon().scheme("file").build() : uri;
            // mediapipe:oss-insert-begin(oss-only)
            // try (InputStream is = context.getContentResolver().openInputStream(safeUri)) {
            //   bytes = ByteStreams.toByteArray(is);
            // }
            // mediapipe:oss-insert-end
          }
        }
      }
    } catch (MediaPipeException e) {
      throw e;
    } catch (FileNotFoundException e) {
      MediaPipeException ex =
          new MediaPipeException(
              MediaPipeException.StatusCode.NOT_FOUND.ordinal(),
              "URI target not found: " + uri + ": " + e.getMessage());
      ex.initCause(e);
      throw ex;
    } catch (IOException | RuntimeException e) {
      MediaPipeException ex =
          new MediaPipeException(
              MediaPipeException.StatusCode.INTERNAL.ordinal(),
              "Eerror loading image from URI: " + uri + ": " + e.getMessage());
      ex.initCause(e);
      throw ex;
    }
    return bytes;
  }
}
