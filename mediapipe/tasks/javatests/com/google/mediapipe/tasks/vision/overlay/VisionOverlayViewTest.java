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

import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link VisionOverlayView} setters used from Compose {@code AndroidView}. */
@RunWith(AndroidJUnit4.class)
public final class VisionOverlayViewTest {

  @Test
  public void constructsWithTransparentBackground() {
    VisionOverlayView view =
        new VisionOverlayView(ApplicationProvider.getApplicationContext());
    assertThat(view.isMirrored()).isFalse();
    assertThat(view.getRotationDegrees()).isEqualTo(0);
    assertThat(view.getStyle()).isNotNull();
  }

  @Test
  public void setMirrored_roundTrips() {
    VisionOverlayView view =
        new VisionOverlayView(ApplicationProvider.getApplicationContext());
    view.setMirrored(true);
    assertThat(view.isMirrored()).isTrue();
  }

  @Test
  public void setRotationDegrees_normalizes() {
    VisionOverlayView view =
        new VisionOverlayView(ApplicationProvider.getApplicationContext());
    view.setRotationDegrees(-90);
    assertThat(view.getRotationDegrees()).isEqualTo(270);
  }

  @Test
  public void setHandLandmarkerResult_rejectsNonPositiveImageSize() {
    VisionOverlayView view =
        new VisionOverlayView(ApplicationProvider.getApplicationContext());
    IllegalArgumentException thrown =
        assertThrows(
            IllegalArgumentException.class,
            () -> view.setHandLandmarkerResult(null, 0, 100, RunningMode.LIVE_STREAM));
    assertThat(thrown).hasMessageThat().contains("Image size must be positive");
  }

  @Test
  public void setHandLandmarkerResult_rejectsNullRunningMode() {
    VisionOverlayView view =
        new VisionOverlayView(ApplicationProvider.getApplicationContext());
    assertThrows(
        IllegalArgumentException.class,
        () -> view.setHandLandmarkerResult(null, 100, 100, null));
  }

  @Test
  public void setStyle_rejectsNull() {
    VisionOverlayView view =
        new VisionOverlayView(ApplicationProvider.getApplicationContext());
    assertThrows(IllegalArgumentException.class, () -> view.setStyle(null));
  }

  @Test
  public void clear_andNullResult_doNotThrow() {
    VisionOverlayView view =
        new VisionOverlayView(ApplicationProvider.getApplicationContext());
    view.setHandLandmarkerResult(null, 128, 128, RunningMode.IMAGE);
    view.setPoseLandmarkerResult(null, 128, 128, RunningMode.VIDEO);
    view.setFaceLandmarkerResult(null, 128, 128, RunningMode.IMAGE);
    view.setObjectDetectorResult(null, 128, 128, RunningMode.LIVE_STREAM);
    view.setFaceDetectorResult(null, 128, 128, RunningMode.IMAGE);
    view.setGestureRecognizerResult(null, 128, 128, RunningMode.IMAGE);
    view.setHolisticLandmarkerResult(null, 128, 128, RunningMode.IMAGE);
    view.clear();
  }
}
