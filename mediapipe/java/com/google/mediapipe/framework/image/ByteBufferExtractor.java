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

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.os.Build.VERSION;
import android.os.Build.VERSION_CODES;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.framework.image.MPImage.MPImageFormat;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;

/**
 * Utility for extracting {@link ByteBuffer} from {@link MPImage}.
 *
 * <p>Currently it only supports {@link MPImage} with {@link MPImage#STORAGE_TYPE_BYTEBUFFER},
 * otherwise {@link IllegalArgumentException} will be thrown.
 */
public class ByteBufferExtractor {

  /**
   * Extracts a {@link ByteBuffer} from a {@link MPImage}.
   *
   * <p>The returned {@link ByteBuffer} is a read-only view, with the first available {@link
   * MPImageProperties} whose storage type is {@code MPImage.STORAGE_TYPE_BYTEBUFFER}.
   *
   * @see MPImage#getContainedImageProperties()
   * @return A read-only {@link ByteBuffer}.
   * @throws IllegalArgumentException when the image doesn't contain a {@link ByteBuffer} storage.
   */
  @SuppressLint("SwitchIntDef")
  public static ByteBuffer extract(MPImage image) {
    MPImageContainer container = image.getContainer();
    switch (container.getImageProperties().getStorageType()) {
      case MPImage.STORAGE_TYPE_BYTEBUFFER:
        ByteBufferImageContainer byteBufferImageContainer = (ByteBufferImageContainer) container;
        return byteBufferImageContainer
            .getByteBuffer()
            .asReadOnlyBuffer()
            .order(ByteOrder.nativeOrder());
      default:
        throw new IllegalArgumentException(
            "Extract ByteBuffer from a MPImage created by objects other than Bytebuffer is not"
                + " supported");
    }
  }

  /**
   * Extracts a readonly {@link ByteBuffer} in given {@code targetFormat} from a {@link MPImage}.
   *
   * <p>Format conversion spec:
   *
   * <ul>
   *   <li>When extracting RGB images to RGBA format, A channel will always set to 255.
   *   <li>When extracting RGBA images to RGB format, A channel will be dropped.
   * </ul>
   *
   * @param image the image to extract buffer from.
   * @param targetFormat the image format of the result bytebuffer.
   * @return the readonly {@link ByteBuffer} stored in {@link MPImage}
   * @throws IllegalArgumentException when the extraction requires unsupported format or data type
   *     conversions.
   */
  public static ByteBuffer extract(MPImage image, @MPImageFormat int targetFormat) {
    MPImageContainer container;
    MPImageProperties byteBufferProperties =
        MPImageProperties.builder()
            .setStorageType(MPImage.STORAGE_TYPE_BYTEBUFFER)
            .setImageFormat(targetFormat)
            .build();
    if ((container = image.getContainer(byteBufferProperties)) != null) {
      ByteBufferImageContainer byteBufferImageContainer = (ByteBufferImageContainer) container;
      return byteBufferImageContainer
          .getByteBuffer()
          .asReadOnlyBuffer()
          .order(ByteOrder.nativeOrder());
    } else if ((container = image.getContainer(MPImage.STORAGE_TYPE_BYTEBUFFER)) != null) {
      ByteBufferImageContainer byteBufferImageContainer = (ByteBufferImageContainer) container;
      @MPImageFormat int sourceFormat = byteBufferImageContainer.getImageFormat();
      return convertByteBuffer(byteBufferImageContainer.getByteBuffer(), sourceFormat, targetFormat)
          .asReadOnlyBuffer()
          .order(ByteOrder.nativeOrder());
    } else if ((container = image.getContainer(MPImage.STORAGE_TYPE_BITMAP)) != null) {
      BitmapImageContainer bitmapImageContainer = (BitmapImageContainer) container;
      ByteBuffer byteBuffer =
          extractByteBufferFromBitmap(bitmapImageContainer.getBitmap(), targetFormat)
              .asReadOnlyBuffer();
      boolean unused = image.addContainer(new ByteBufferImageContainer(byteBuffer, targetFormat));
      return byteBuffer;
    } else {
      throw new IllegalArgumentException(
          "Extracting ByteBuffer from a MPImage created by objects other than Bitmap or"
              + " Bytebuffer is not supported");
    }
  }

  /** A wrapper for a {@link ByteBuffer} and its {@link MPImageFormat}. */
  @AutoValue
  abstract static class Result {
    /**
     * Gets the {@link ByteBuffer} in the result of {@link ByteBufferExtractor#extract(MPImage)}.
     */
    public abstract ByteBuffer buffer();

    /**
     * Gets the {@link MPImageFormat} in the result of {@link ByteBufferExtractor#extract(MPImage)}.
     */
    @MPImageFormat
    public abstract int format();

    static Result create(ByteBuffer buffer, @MPImageFormat int imageFormat) {
      return new AutoValue_ByteBufferExtractor_Result(buffer, imageFormat);
    }
  }

  /**
   * Extracts a {@link ByteBuffer} in any available {@code imageFormat} from a {@link MPImage}.
   *
   * <p>It will make the best effort to return an already existed {@link ByteBuffer} to avoid copy.
   *
   * @return the readonly {@link ByteBuffer} stored in {@link MPImage}
   * @throws IllegalArgumentException when {@code image} doesn't contain {@link ByteBuffer} with
   *     given {@code imageFormat}
   */
  static Result extractInRecommendedFormat(MPImage image) {
    MPImageContainer container;
    if ((container = image.getContainer(MPImage.STORAGE_TYPE_BITMAP)) != null) {
      Bitmap bitmap = ((BitmapImageContainer) container).getBitmap();
      @MPImageFormat int format = adviseImageFormat(bitmap);
      Result result =
          Result.create(extractByteBufferFromBitmap(bitmap, format).asReadOnlyBuffer(), format);

      boolean unused =
          image.addContainer(new ByteBufferImageContainer(result.buffer(), result.format()));
      return result;
    } else if ((container = image.getContainer(MPImage.STORAGE_TYPE_BYTEBUFFER)) != null) {
      ByteBufferImageContainer byteBufferImageContainer = (ByteBufferImageContainer) container;
      return Result.create(
          byteBufferImageContainer.getByteBuffer().asReadOnlyBuffer(),
          byteBufferImageContainer.getImageFormat());
    } else {
      throw new IllegalArgumentException(
          "Extract ByteBuffer from a MPImage created by objects other than Bitmap or Bytebuffer"
              + " is not supported");
    }
  }

  @MPImageFormat
  private static int adviseImageFormat(Bitmap bitmap) {
    if (bitmap.getConfig() == Config.ARGB_8888) {
      return MPImage.IMAGE_FORMAT_RGBA;
    } else {
      throw new IllegalArgumentException(
          String.format(
              "Extracting ByteBuffer from a MPImage created by a Bitmap in config %s is not"
                  + " supported",
              bitmap.getConfig()));
    }
  }

  private static ByteBuffer extractByteBufferFromBitmap(
      Bitmap bitmap, @MPImageFormat int imageFormat) {
    if (VERSION.SDK_INT >= VERSION_CODES.JELLY_BEAN_MR1 && bitmap.isPremultiplied()) {
      throw new IllegalArgumentException(
          "Extracting ByteBuffer from a MPImage created by a premultiplied Bitmap is not"
              + " supported");
    }
    if (bitmap.getConfig() == Config.ARGB_8888) {
      if (imageFormat == MPImage.IMAGE_FORMAT_RGBA) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(bitmap.getByteCount());
        bitmap.copyPixelsToBuffer(buffer);
        buffer.rewind();
        return buffer;
      } else if (imageFormat == MPImage.IMAGE_FORMAT_RGB) {
        // TODO: Try Use RGBA buffer to create RGB buffer which might be faster.
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();
        int[] pixels = new int[w * h];
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h);
        ByteBuffer buffer = ByteBuffer.allocateDirect(w * h * 3);
        buffer.order(ByteOrder.nativeOrder());
        for (int pixel : pixels) {
          // getPixels returns Color in ARGB rather than copyPixelsToBuffer which returns RGBA
          buffer.put((byte) ((pixel >> 16) & 0xff));
          buffer.put((byte) ((pixel >> 8) & 0xff));
          buffer.put((byte) (pixel & 0xff));
        }
        buffer.rewind();
        return buffer;
      }
    }
    throw new IllegalArgumentException(
        String.format(
            "Extracting ByteBuffer from a MPImage created by Bitmap and convert from %s to format"
                + " %d is not supported",
            bitmap.getConfig(), imageFormat));
  }

  private static ByteBuffer convertByteBuffer(
      ByteBuffer source, @MPImageFormat int sourceFormat, @MPImageFormat int targetFormat) {
    if (sourceFormat == MPImage.IMAGE_FORMAT_RGB && targetFormat == MPImage.IMAGE_FORMAT_RGBA) {
      ByteBuffer target = ByteBuffer.allocateDirect(source.capacity() / 3 * 4);
      // Extend the buffer when the target is longer than the source. Use two cursors and sweep the
      // array reversely to convert in-place.
      byte[] array = new byte[target.capacity()];
      source.get(array, 0, source.capacity());
      source.rewind();
      int rgbCursor = source.capacity();
      int rgbaCursor = target.capacity();
      while (rgbCursor != rgbaCursor) {
        array[--rgbaCursor] = (byte) 0xff; // A
        array[--rgbaCursor] = array[--rgbCursor]; // B
        array[--rgbaCursor] = array[--rgbCursor]; // G
        array[--rgbaCursor] = array[--rgbCursor]; // R
      }
      target.put(array, 0, target.capacity());
      target.rewind();
      return target;
    } else if (sourceFormat == MPImage.IMAGE_FORMAT_RGBA
        && targetFormat == MPImage.IMAGE_FORMAT_RGB) {
      ByteBuffer target = ByteBuffer.allocateDirect(source.capacity() / 4 * 3);
      // Shrink the buffer when the target is shorter than the source. Use two cursors and sweep the
      // array to convert in-place.
      byte[] array = new byte[source.capacity()];
      source.get(array, 0, source.capacity());
      source.rewind();
      int rgbaCursor = 0;
      int rgbCursor = 0;
      while (rgbaCursor < array.length) {
        array[rgbCursor++] = array[rgbaCursor++]; // R
        array[rgbCursor++] = array[rgbaCursor++]; // G
        array[rgbCursor++] = array[rgbaCursor++]; // B
        rgbaCursor++;
      }
      target.put(array, 0, target.capacity());
      target.rewind();
      return target;
    } else {
      throw new IllegalArgumentException(
          String.format(
              Locale.ENGLISH,
              "Convert bytebuffer image format from %d to %d is not supported",
              sourceFormat,
              targetFormat));
    }
  }

  // ByteBuffer is not able to be instantiated.
  private ByteBufferExtractor() {}
}
