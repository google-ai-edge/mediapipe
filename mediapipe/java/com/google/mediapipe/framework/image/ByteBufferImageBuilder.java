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

import com.google.mediapipe.framework.image.MPImage.MPImageFormat;
import java.nio.ByteBuffer;

/**
 * Builds a {@link MPImage} from a {@link ByteBuffer}.
 *
 * <p>You can pass in either mutable or immutable {@link ByteBuffer}. However once {@link
 * ByteBuffer} is passed in, to keep data integrity you shouldn't modify content in it.
 *
 * <p>Use {@link ByteBufferExtractor} to get {@link ByteBuffer} you passed in.
 */
public class ByteBufferImageBuilder {

  // Mandatory fields.
  private final ByteBuffer buffer;
  private final int width;
  private final int height;
  @MPImageFormat private final int imageFormat;

  // Optional fields.
  private long timestamp;

  /**
   * Creates the builder with mandatory {@link ByteBuffer} and the represented image.
   *
   * <p>We will validate the size of the {@code byteBuffer} with given {@code width}, {@code height}
   * and {@code imageFormat}.
   *
   * @param byteBuffer image data object.
   * @param width the width of the represented image.
   * @param height the height of the represented image.
   * @param imageFormat how the data encode the image.
   */
  public ByteBufferImageBuilder(
      ByteBuffer byteBuffer, int width, int height, @MPImageFormat int imageFormat) {
    this.buffer = byteBuffer;
    this.width = width;
    this.height = height;
    this.imageFormat = imageFormat;
    // TODO: Validate bytebuffer size with width, height and image format
    this.timestamp = 0;
  }

  /** Sets value for {@link MPImage#getTimestamp()}. */
  ByteBufferImageBuilder setTimestamp(long timestamp) {
    this.timestamp = timestamp;
    return this;
  }

  /** Builds a {@link MPImage} instance. */
  public MPImage build() {
    return new MPImage(new ByteBufferImageContainer(buffer, imageFormat), timestamp, width, height);
  }
}
