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

import com.google.mediapipe.framework.image.Image.ImageFormat;
import java.nio.ByteBuffer;

class ByteBufferImageContainer implements ImageContainer {

  private final ByteBuffer buffer;
  private final ImageProperties properties;

  public ByteBufferImageContainer(
      ByteBuffer buffer,
      @ImageFormat int imageFormat) {
    this.buffer = buffer;
    this.properties =
        ImageProperties.builder()
            .setStorageType(Image.STORAGE_TYPE_BYTEBUFFER)
            .setImageFormat(imageFormat)
            .build();
  }

  public ByteBuffer getByteBuffer() {
    return buffer;
  }

  @Override
  public ImageProperties getImageProperties() {
    return properties;
  }

  /**
   * Returns the image format.
   */
  @ImageFormat
  public int getImageFormat() {
    return properties.getImageFormat();
  }

  @Override
  public void close() {
    // No op for ByteBuffer.
  }
}
