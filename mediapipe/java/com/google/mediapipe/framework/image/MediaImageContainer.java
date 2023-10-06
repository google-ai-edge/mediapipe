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

import android.media.Image;
import android.os.Build;
import android.os.Build.VERSION;
import android.os.Build.VERSION_CODES;
import androidx.annotation.RequiresApi;
import com.google.mediapipe.framework.image.MPImage.MPImageFormat;

@RequiresApi(VERSION_CODES.KITKAT)
class MediaImageContainer implements MPImageContainer {

  private final Image mediaImage;
  private final MPImageProperties properties;

  public MediaImageContainer(Image mediaImage) {
    this.mediaImage = mediaImage;
    this.properties =
        MPImageProperties.builder()
            .setStorageType(MPImage.STORAGE_TYPE_MEDIA_IMAGE)
            .setImageFormat(convertFormatCode(mediaImage.getFormat()))
            .build();
  }

  public Image getImage() {
    return mediaImage;
  }

  @Override
  public MPImageProperties getImageProperties() {
    return properties;
  }

  @Override
  public void close() {
    mediaImage.close();
  }

  @MPImageFormat
  static int convertFormatCode(int graphicsFormat) {
    // We only cover the format mentioned in
    // https://developer.android.com/reference/android/media/Image#getFormat()
    if (VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (graphicsFormat == android.graphics.ImageFormat.FLEX_RGBA_8888) {
        return MPImage.IMAGE_FORMAT_RGBA;
      } else if (graphicsFormat == android.graphics.ImageFormat.FLEX_RGB_888) {
        return MPImage.IMAGE_FORMAT_RGB;
      }
    }
    switch (graphicsFormat) {
      case android.graphics.ImageFormat.JPEG:
        return MPImage.IMAGE_FORMAT_JPEG;
      case android.graphics.ImageFormat.YUV_420_888:
        return MPImage.IMAGE_FORMAT_YUV_420_888;
      default:
        return MPImage.IMAGE_FORMAT_UNKNOWN;
    }
  }
}
