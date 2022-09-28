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
 * Utility for extracting {@link android.media.Image} from {@link Image}.
 *
 * <p>Currently it only supports {@link Image} with {@link Image#STORAGE_TYPE_MEDIA_IMAGE},
 * otherwise {@link IllegalArgumentException} will be thrown.
 */
@RequiresApi(VERSION_CODES.KITKAT)
public class MediaImageExtractor {

  private MediaImageExtractor() {}

  /**
   * Extracts a {@link android.media.Image} from an {@link Image}. Currently it only works for
   * {@link Image} that built from {@link MediaImageBuilder}.
   *
   * @param image the image to extract {@link android.media.Image} from.
   * @return {@link android.media.Image} that stored in {@link Image}.
   * @throws IllegalArgumentException if the extraction failed.
   */
  public static android.media.Image extract(Image image) {
    ImageContainer container;
    if ((container = image.getContainer(Image.STORAGE_TYPE_MEDIA_IMAGE)) != null) {
      return ((MediaImageContainer) container).getImage();
    }
    throw new IllegalArgumentException(
        "Extract Media Image from an Image created by objects other than Media Image"
            + " is not supported");
  }
}
