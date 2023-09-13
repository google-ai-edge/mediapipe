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
import android.os.Build.VERSION_CODES;
import androidx.annotation.RequiresApi;

/**
 * Builds {@link MPImage} from {@link android.media.Image}.
 *
 * <p>Once {@link android.media.Image} is passed in, to keep data integrity you shouldn't modify
 * content in it.
 *
 * <p>Use {@link MediaImageExtractor} to get {@link android.media.Image} you passed in.
 */
@RequiresApi(VERSION_CODES.KITKAT)
public class MediaImageBuilder {

  // Mandatory fields.
  private final Image mediaImage;

  // Optional fields.
  private long timestamp;

  /**
   * Creates the builder with a mandatory {@link android.media.Image}.
   *
   * @param mediaImage image data object.
   */
  public MediaImageBuilder(Image mediaImage) {
    this.mediaImage = mediaImage;
    this.timestamp = 0;
  }

  /** Sets value for {@link MPImage#getTimestamp()}. */
  MediaImageBuilder setTimestamp(long timestamp) {
    this.timestamp = timestamp;
    return this;
  }

  /** Builds a {@link MPImage} instance. */
  public MPImage build() {
    return new MPImage(
        new MediaImageContainer(mediaImage),
        timestamp,
        mediaImage.getWidth(),
        mediaImage.getHeight());
  }
}
