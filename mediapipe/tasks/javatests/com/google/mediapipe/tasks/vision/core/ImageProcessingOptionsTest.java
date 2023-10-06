// Copyright 2022 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.tasks.vision.core;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import android.graphics.RectF;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Test for {@link ImageProcessingOptions}/ */
@RunWith(AndroidJUnit4.class)
public final class ImageProcessingOptionsTest {

  @Test
  public void succeedsWithValidInputs() throws Exception {
    ImageProcessingOptions options =
        ImageProcessingOptions.builder()
            .setRegionOfInterest(new RectF(0.0f, 0.1f, 1.0f, 0.9f))
            .setRotationDegrees(270)
            .build();
  }

  @Test
  public void failsWithLeftHigherThanRight() {
    IllegalArgumentException exception =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageProcessingOptions.builder()
                    .setRegionOfInterest(new RectF(0.9f, 0.0f, 0.1f, 1.0f))
                    .build());
    assertThat(exception).hasMessageThat().contains("Expected left < right and top < bottom");
  }

  @Test
  public void failsWithBottomHigherThanTop() {
    IllegalArgumentException exception =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageProcessingOptions.builder()
                    .setRegionOfInterest(new RectF(0.0f, 0.9f, 1.0f, 0.1f))
                    .build());
    assertThat(exception).hasMessageThat().contains("Expected left < right and top < bottom");
  }

  @Test
  public void failsWithInvalidRotation() {
    IllegalArgumentException exception =
        assertThrows(
            IllegalArgumentException.class,
            () -> ImageProcessingOptions.builder().setRotationDegrees(1).build());
    assertThat(exception).hasMessageThat().contains("Expected rotation to be a multiple of 90°");
  }
}
