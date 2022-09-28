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

import android.os.Build.VERSION_CODES;
import androidx.annotation.RequiresApi;

/**
 * Builds {@link Image} from {@link android.media.Image}.
 *
 * <p>Once {@link android.media.Image} is passed in, to keep data integrity you shouldn't modify
 * content in it.
 *
 * <p>Use {@link MediaImageExtractor} to get {@link android.media.Image} you passed in.
 */
@RequiresApi(VERSION_CODES.KITKAT)
public class MediaImageBuilder {

  // Mandatory fields.
  private final android.media.Image mediaImage;

  // Optional fields.
  private long timestamp;

  /**
   * Creates the builder with a mandatory {@link android.media.Image}.
   *
   * @param mediaImage image data object.
   */
  public MediaImageBuilder(android.media.Image mediaImage) {
    this.mediaImage = mediaImage;
    this.timestamp = 0;
  }

  /** Sets value for {@link Image#getTimestamp()}. */
  MediaImageBuilder setTimestamp(long timestamp) {
    this.timestamp = timestamp;
    return this;
  }

  /** Builds an {@link Image} instance. */
  public Image build() {
    return new Image(
        new MediaImageContainer(mediaImage),
        timestamp,
        mediaImage.getWidth(),
        mediaImage.getHeight());
  }
}
