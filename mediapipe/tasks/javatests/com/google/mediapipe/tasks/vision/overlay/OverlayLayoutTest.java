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

import android.graphics.RectF;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link OverlayLayout} — FIT vs FILL, mirroring, and camera-frame rotation. */
@RunWith(AndroidJUnit4.class)
public final class OverlayLayoutTest {

  @Test
  public void liveStream_portraitPreviewOnWideCameraFrame() {
    // Typical Phone UI 1080x1920 over a 640x480 camera frame.
    OverlayLayout layout = OverlayLayout.create(1080, 1920, 640, 480, RunningMode.LIVE_STREAM);
    assertThat(layout.scale()).isEqualTo(4.f);
    assertThat(layout.mapX(0.5f)).isEqualTo(1280.f);
    assertThat(layout.mapY(0.5f)).isEqualTo(960.f);
  }

  @Test
  public void imageMode_fitsWithMinScale() {
    // View 200x100, image 100x100 → min(2, 1) = 1 (letterbox, FIT_START).
    OverlayLayout layout =
        OverlayLayout.create(200, 100, 100, 100, RunningMode.IMAGE);
    assertThat(layout.scale()).isEqualTo(1.f);
    assertThat(layout.mapX(0.5f)).isEqualTo(50.f);
    assertThat(layout.mapY(0.5f)).isEqualTo(50.f);
  }

  @Test
  public void videoMode_matchesImageFit() {
    OverlayLayout image = OverlayLayout.create(200, 100, 100, 100, RunningMode.IMAGE);
    OverlayLayout video = OverlayLayout.create(200, 100, 100, 100, RunningMode.VIDEO);
    assertThat(video.scale()).isEqualTo(image.scale());
  }

  @Test
  public void liveStream_fillsWithMaxScale() {
    // View 200x100, image 100x100 → max(2, 1) = 2 (crop, FILL_START / PreviewView).
    OverlayLayout layout =
        OverlayLayout.create(200, 100, 100, 100, RunningMode.LIVE_STREAM);
    assertThat(layout.scale()).isEqualTo(2.f);
    assertThat(layout.mapX(0.5f)).isEqualTo(100.f);
    assertThat(layout.mapY(0.5f)).isEqualTo(100.f);
  }

  @Test
  public void equalViewAndImage_scaleIsOne() {
    OverlayLayout layout = OverlayLayout.create(128, 128, 128, 128, RunningMode.IMAGE);
    assertThat(layout.scale()).isEqualTo(1.f);
    assertThat(layout.mapX(0.f)).isEqualTo(0.f);
    assertThat(layout.mapX(1.f)).isEqualTo(128.f);
    assertThat(layout.mapY(1.f)).isEqualTo(128.f);
  }

  @Test
  public void mirrored_flipsNormalizedX() {
    OverlayLayout layout =
        OverlayLayout.create(
            100, 100, 100, 100, RunningMode.IMAGE, /* rotationDegrees= */ 0, /* mirrored= */ true);
    assertThat(layout.mirrored()).isTrue();
    // x=0.25 → 0.75 * 100 = 75
    assertThat(layout.mapX(0.25f)).isEqualTo(75.f);
    assertThat(layout.mapX(0.f)).isEqualTo(100.f);
    assertThat(layout.mapX(1.f)).isEqualTo(0.f);
    assertThat(layout.mapY(0.25f)).isEqualTo(25.f);
  }

  @Test
  public void rotation90_swapsImageSizeForScale() {
    OverlayLayout layout =
        OverlayLayout.create(
            200, 200, /* imageWidth= */ 100, /* imageHeight= */ 200, RunningMode.IMAGE, 90, false);
    assertThat(layout.rotationDegrees()).isEqualTo(90);
    assertThat(layout.rotatedImageWidth()).isEqualTo(200);
    assertThat(layout.rotatedImageHeight()).isEqualTo(100);
    // min(200/200, 200/100) = min(1, 2) = 1
    assertThat(layout.scale()).isEqualTo(1.f);
  }

  @Test
  public void rotation270_sameSwapAs90() {
    OverlayLayout layout =
        OverlayLayout.create(200, 200, 100, 200, RunningMode.IMAGE, 270, false);
    assertThat(layout.rotatedImageWidth()).isEqualTo(200);
    assertThat(layout.rotatedImageHeight()).isEqualTo(100);
  }

  @Test
  public void negativeRotation_normalizes() {
    OverlayLayout layout =
        OverlayLayout.create(100, 100, 100, 100, RunningMode.IMAGE, -90, false);
    assertThat(layout.rotationDegrees()).isEqualTo(270);
  }

  @Test
  public void rotation450_normalizesTo90() {
    OverlayLayout layout =
        OverlayLayout.create(100, 100, 100, 100, RunningMode.IMAGE, 450, false);
    assertThat(layout.rotationDegrees()).isEqualTo(90);
  }

  @Test
  public void mapImageRect_noRotation_scales() {
    OverlayLayout layout = OverlayLayout.create(200, 200, 100, 100, RunningMode.IMAGE);
    RectF mapped = layout.mapImageRect(new RectF(10.f, 20.f, 30.f, 40.f));
    assertThat(mapped.left).isEqualTo(20.f);
    assertThat(mapped.top).isEqualTo(40.f);
    assertThat(mapped.right).isEqualTo(60.f);
    assertThat(mapped.bottom).isEqualTo(80.f);
  }

  @Test
  public void mapImageRect_rotation180_sendsTopLeftToBottomRight() {
    OverlayLayout layout =
        OverlayLayout.create(100, 100, 100, 100, RunningMode.IMAGE, 180, false);
    RectF mapped = layout.mapImageRect(new RectF(0.f, 0.f, 10.f, 10.f));
    assertThat(mapped.left).isEqualTo(90.f);
    assertThat(mapped.top).isEqualTo(90.f);
    assertThat(mapped.right).isEqualTo(100.f);
    assertThat(mapped.bottom).isEqualTo(100.f);
  }

  @Test
  public void mapImageRect_rotation90_fullFrameCoversRotatedBounds() {
    OverlayLayout layout =
        OverlayLayout.create(200, 100, 100, 200, RunningMode.IMAGE, 90, false);
    RectF mapped = layout.mapImageRect(new RectF(0.f, 0.f, 100.f, 200.f));
    assertThat(mapped.left).isEqualTo(0.f);
    assertThat(mapped.top).isEqualTo(0.f);
    assertThat(mapped.right).isEqualTo(200.f);
    assertThat(mapped.bottom).isEqualTo(100.f);
  }

  @Test
  public void mapImageRect_mirrored_flipsHorizontally() {
    OverlayLayout layout =
        OverlayLayout.create(100, 100, 100, 100, RunningMode.IMAGE, 0, true);
    RectF mapped = layout.mapImageRect(new RectF(0.f, 10.f, 20.f, 30.f));
    assertThat(mapped.left).isEqualTo(80.f);
    assertThat(mapped.right).isEqualTo(100.f);
    assertThat(mapped.top).isEqualTo(10.f);
    assertThat(mapped.bottom).isEqualTo(30.f);
  }

  @Test
  public void rejectsNonPositiveViewSize() {
    IllegalArgumentException thrown =
        assertThrows(
            IllegalArgumentException.class,
            () -> OverlayLayout.create(0, 100, 100, 100, RunningMode.IMAGE));
    assertThat(thrown).hasMessageThat().contains("View size must be positive");
  }

  @Test
  public void rejectsNonPositiveImageSize() {
    IllegalArgumentException thrown =
        assertThrows(
            IllegalArgumentException.class,
            () -> OverlayLayout.create(100, 100, 100, 0, RunningMode.IMAGE));
    assertThat(thrown).hasMessageThat().contains("Image size must be positive");
  }

  @Test
  public void rejectsNon90MultipleRotation() {
    IllegalArgumentException thrown =
        assertThrows(
            IllegalArgumentException.class,
            () -> OverlayLayout.create(100, 100, 100, 100, RunningMode.IMAGE, 45, false));
    assertThat(thrown).hasMessageThat().contains("multiple of 90");
  }

  @Test
  public void mapImageRect_nullBox_throws() {
    OverlayLayout layout = OverlayLayout.create(100, 100, 100, 100, RunningMode.IMAGE);
    assertThrows(IllegalArgumentException.class, () -> layout.mapImageRect(null));
  }

  @Test
  public void normalizeRotation_rejectsNonMultiples() {
    assertThrows(IllegalArgumentException.class, () -> OverlayLayout.normalizeRotation(1));
  }
}
