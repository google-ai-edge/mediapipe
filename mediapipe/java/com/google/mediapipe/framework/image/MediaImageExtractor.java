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
 * Utility for extracting {@link android.media.Image} from {@link MPImage}.
 *
 * <p>Currently it only supports {@link MPImage} with {@link MPImage#STORAGE_TYPE_MEDIA_IMAGE},
 * otherwise {@link IllegalArgumentException} will be thrown.
 */
@RequiresApi(VERSION_CODES.KITKAT)
public class MediaImageExtractor {

  private MediaImageExtractor() {}

  /**
   * Extracts a {@link android.media.Image} from a {@link MPImage}. Currently it only works for
   * {@link MPImage} that built from {@link MediaImageBuilder}.
   *
   * @param image the image to extract {@link android.media.Image} from.
   * @return {@link android.media.Image} that stored in {@link MPImage}.
   * @throws IllegalArgumentException if the extraction failed.
   */
  public static Image extract(MPImage image) {
    MPImageContainer container;
    if ((container = image.getContainer(MPImage.STORAGE_TYPE_MEDIA_IMAGE)) != null) {
      return ((MediaImageContainer) container).getImage();
    }
    throw new IllegalArgumentException(
        "Extract Media Image from a MPImage created by objects other than Media Image"
            + " is not supported");
  }
}
