// Copyright 2026 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.vision.overlay;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import android.graphics.Color;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link OverlayStyle}. */
@RunWith(AndroidJUnit4.class)
public final class OverlayStyleTest {

  @Test
  public void mediapipeDefault_scalesWithDensity() {
    OverlayStyle mdpi = OverlayStyle.mediapipeDefault(1.f);
    OverlayStyle xhdpi = OverlayStyle.mediapipeDefault(2.f);
    assertThat(xhdpi.landmarkRadius()).isEqualTo(mdpi.landmarkRadius() * 2.f);
    assertThat(xhdpi.connectionStrokeWidth()).isEqualTo(mdpi.connectionStrokeWidth() * 2.f);
    assertThat(xhdpi.textSize()).isEqualTo(mdpi.textSize() * 2.f);
    assertThat(xhdpi.boxStrokeWidth()).isEqualTo(mdpi.boxStrokeWidth() * 2.f);
  }

  @Test
  public void mediapipeDefault_rejectsNonPositiveDensity() {
    assertThrows(IllegalArgumentException.class, () -> OverlayStyle.mediapipeDefault(0.f));
    assertThrows(IllegalArgumentException.class, () -> OverlayStyle.mediapipeDefault(-1.f));
  }

  @Test
  public void builder_rejectsNonPositiveRadius() {
    assertThrows(
        IllegalArgumentException.class, () -> OverlayStyle.builder().setLandmarkRadius(0.f));
  }

  @Test
  public void builder_rejectsNegativePadding() {
    assertThrows(
        IllegalArgumentException.class, () -> OverlayStyle.builder().setTextPadding(-1.f));
  }

  @Test
  public void toBuilder_roundTrips() {
    OverlayStyle original =
        OverlayStyle.builder()
            .setLandmarkColor(Color.RED)
            .setConnectionColor(Color.BLUE)
            .setBoxColor(Color.GREEN)
            .setTextSize(12.f)
            .build();
    OverlayStyle copy = original.toBuilder().build();
    assertThat(copy.landmarkColor()).isEqualTo(Color.RED);
    assertThat(copy.connectionColor()).isEqualTo(Color.BLUE);
    assertThat(copy.boxColor()).isEqualTo(Color.GREEN);
    assertThat(copy.textSize()).isEqualTo(12.f);
  }

  @Test
  public void defaultColors_areOpaque() {
    OverlayStyle style = OverlayStyle.mediapipeDefault();
    assertThat(Color.alpha(style.landmarkColor())).isEqualTo(255);
    assertThat(Color.alpha(style.connectionColor())).isEqualTo(255);
    assertThat(Color.alpha(style.boxColor())).isEqualTo(255);
  }
}
