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

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.RectF;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.Connection;
import com.google.mediapipe.tasks.components.containers.Detection;
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link VisionOverlayRenderer} drawing onto a bitmap. */
@RunWith(AndroidJUnit4.class)
public final class VisionOverlayRendererTest {

  private static final int SIZE = 100;
  private static final OverlayLayout LAYOUT =
      OverlayLayout.create(SIZE, SIZE, SIZE, SIZE, RunningMode.IMAGE);

  @Test
  public void drawLandmarks_nullConnections_stillDrawsPoints() {
    OverlayStyle style =
        OverlayStyle.builder().setLandmarkColor(Color.RED).setLandmarkRadius(12.f).build();
    Bitmap bitmap = emptyBitmap();
    List<List<NormalizedLandmark>> landmarks =
        Collections.singletonList(
            Collections.singletonList(NormalizedLandmark.create(0.5f, 0.5f, 0.f)));

    VisionOverlayRenderer.drawLandmarks(
        new Canvas(bitmap), landmarks, /* connections= */ null, LAYOUT, style);

    assertPixelCloseTo(bitmap, 50, 50, Color.RED);
  }

  @Test
  public void drawLandmarks_paintsCenterPixel() {
    OverlayStyle style =
        OverlayStyle.builder().setLandmarkColor(Color.RED).setLandmarkRadius(12.f).build();
    Bitmap bitmap = emptyBitmap();
    List<List<NormalizedLandmark>> landmarks =
        Collections.singletonList(
            Collections.singletonList(NormalizedLandmark.create(0.5f, 0.5f, 0.f)));

    VisionOverlayRenderer.drawLandmarks(
        new Canvas(bitmap), landmarks, Collections.emptySet(), LAYOUT, style);

    assertPixelCloseTo(bitmap, 50, 50, Color.RED);
  }

  @Test
  public void drawLandmarks_drawsConnectionAcrossMidline() {
    OverlayStyle style =
        OverlayStyle.builder()
            .setConnectionColor(Color.BLUE)
            .setConnectionStrokeWidth(8.f)
            .setLandmarkRadius(1.f)
            .setLandmarkColor(Color.TRANSPARENT)
            .build();
    Bitmap bitmap = emptyBitmap();
    List<NormalizedLandmark> pair =
        Arrays.asList(
            NormalizedLandmark.create(0.1f, 0.5f, 0.f), NormalizedLandmark.create(0.9f, 0.5f, 0.f));
    Set<Connection> connections = Collections.singleton(Connection.create(0, 1));

    VisionOverlayRenderer.drawLandmarks(
        new Canvas(bitmap), Collections.singletonList(pair), connections, LAYOUT, style);

    assertPixelCloseTo(bitmap, 50, 50, Color.BLUE);
  }

  @Test
  public void drawLandmarks_skipsOutOfRangeConnections() {
    OverlayStyle style =
        OverlayStyle.builder().setLandmarkColor(Color.RED).setLandmarkRadius(10.f).build();
    Bitmap bitmap = emptyBitmap();
    List<NormalizedLandmark> one = Collections.singletonList(NormalizedLandmark.create(0.5f, 0.5f, 0.f));
    Set<Connection> bad = new HashSet<>(Arrays.asList(Connection.create(0, 99), Connection.create(-1, 0)));

    VisionOverlayRenderer.drawLandmarks(
        new Canvas(bitmap), Collections.singletonList(one), bad, LAYOUT, style);

    assertPixelCloseTo(bitmap, 50, 50, Color.RED);
  }

  @Test
  public void drawLandmarks_emptyAndNull_areNoOps() {
    OverlayStyle style = OverlayStyle.mediapipeDefault();
    Bitmap bitmap = emptyBitmap();
    Canvas canvas = new Canvas(bitmap);

    VisionOverlayRenderer.drawLandmarks(canvas, null, Collections.emptySet(), LAYOUT, style);
    VisionOverlayRenderer.drawLandmarks(
        canvas, Collections.emptyList(), Collections.emptySet(), LAYOUT, style);
    VisionOverlayRenderer.drawLandmarks(
        canvas,
        Collections.singletonList(Collections.emptyList()),
        Collections.emptySet(),
        LAYOUT,
        style);

    assertThat(bitmap.getPixel(50, 50)).isEqualTo(Color.TRANSPARENT);
  }

  @Test
  public void drawFaceDetections_nullResult_isNoOp() {
    Bitmap bitmap = emptyBitmap();
    VisionOverlayRenderer.drawFaceDetections(
        new Canvas(bitmap), null, LAYOUT, OverlayStyle.mediapipeDefault());
    assertThat(bitmap.getPixel(0, 0)).isEqualTo(Color.TRANSPARENT);
  }

  @Test
  public void drawGestures_nullResult_isNoOp() {
    Bitmap bitmap = emptyBitmap();
    VisionOverlayRenderer.drawGestures(
        new Canvas(bitmap), null, LAYOUT, OverlayStyle.mediapipeDefault());
    assertThat(bitmap.getPixel(0, 0)).isEqualTo(Color.TRANSPARENT);
  }

  @Test
  public void drawHolistic_nullResult_isNoOp() {
    Bitmap bitmap = emptyBitmap();
    VisionOverlayRenderer.drawHolistic(
        new Canvas(bitmap), null, LAYOUT, OverlayStyle.mediapipeDefault());
    assertThat(bitmap.getPixel(0, 0)).isEqualTo(Color.TRANSPARENT);
  }

  @Test
  public void drawHands_nullResult_isNoOp() {
    Bitmap bitmap = emptyBitmap();
    VisionOverlayRenderer.drawHands(
        new Canvas(bitmap), null, LAYOUT, OverlayStyle.mediapipeDefault());
    assertThat(bitmap.getPixel(0, 0)).isEqualTo(Color.TRANSPARENT);
  }

  @Test
  public void drawPose_nullResult_isNoOp() {
    Bitmap bitmap = emptyBitmap();
    VisionOverlayRenderer.drawPose(
        new Canvas(bitmap), null, LAYOUT, OverlayStyle.mediapipeDefault());
    assertThat(bitmap.getPixel(0, 0)).isEqualTo(Color.TRANSPARENT);
  }

  @Test
  public void drawFaceLandmarks_nullResult_isNoOp() {
    Bitmap bitmap = emptyBitmap();
    VisionOverlayRenderer.drawFaceLandmarks(
        new Canvas(bitmap), null, LAYOUT, OverlayStyle.mediapipeDefault());
    assertThat(bitmap.getPixel(0, 0)).isEqualTo(Color.TRANSPARENT);
  }

  @Test
  public void drawObjects_nullResult_isNoOp() {
    Bitmap bitmap = emptyBitmap();
    VisionOverlayRenderer.drawObjects(
        new Canvas(bitmap), null, LAYOUT, OverlayStyle.mediapipeDefault());
    assertThat(bitmap.getPixel(0, 0)).isEqualTo(Color.TRANSPARENT);
  }

  @Test
  public void drawDetections_strokesBox() {
    OverlayStyle style =
        OverlayStyle.builder().setBoxColor(Color.GREEN).setBoxStrokeWidth(4.f).build();
    Bitmap bitmap = emptyBitmap();
    Detection detection =
        Detection.create(
            Collections.singletonList(Category.create(0.9f, 0, "person", "")),
            new RectF(10.f, 10.f, 90.f, 90.f));

    VisionOverlayRenderer.drawDetections(
        new Canvas(bitmap),
        Collections.singletonList(detection),
        LAYOUT,
        style,
        /* drawKeypoints= */ false);

    // Right edge, away from the label drawn at the top-left of the box.
    assertPixelCloseTo(bitmap, 90, 50, Color.GREEN);
  }

  @Test
  public void drawDetections_drawsKeypointsWhenRequested() {
    OverlayStyle style =
        OverlayStyle.builder().setLandmarkColor(Color.MAGENTA).setLandmarkRadius(8.f).build();
    Bitmap bitmap = emptyBitmap();
    Detection detection =
        Detection.create(
            Collections.singletonList(Category.create(1.f, 0, "face", "Face")),
            new RectF(0.f, 0.f, 10.f, 10.f),
            Optional.of(
                Collections.singletonList(NormalizedKeypoint.create(0.5f, 0.5f))));

    VisionOverlayRenderer.drawDetections(
        new Canvas(bitmap), Collections.singletonList(detection), LAYOUT, style, true);

    assertPixelCloseTo(bitmap, 50, 50, Color.MAGENTA);
  }

  @Test
  public void drawDetections_empty_isNoOp() {
    Bitmap bitmap = emptyBitmap();
    VisionOverlayRenderer.drawDetections(
        new Canvas(bitmap),
        Collections.emptyList(),
        LAYOUT,
        OverlayStyle.mediapipeDefault(),
        false);
    assertThat(bitmap.getPixel(50, 50)).isEqualTo(Color.TRANSPARENT);
  }

  @Test
  public void formatCategory_prefersDisplayName() {
    Category category = Category.create(0.875f, 1, "person", "Person");
    assertThat(VisionOverlayRenderer.formatCategory(category)).isEqualTo("Person 0.88");
  }

  @Test
  public void formatCategory_fallsBackToCategoryName() {
    Category category = Category.create(1.f, 0, "dog", "");
    assertThat(VisionOverlayRenderer.formatCategory(category)).isEqualTo("dog 1.00");
  }

  @Test
  public void rejectsNullCanvas() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            VisionOverlayRenderer.drawLandmarks(
                null,
                Collections.emptyList(),
                Collections.emptySet(),
                LAYOUT,
                OverlayStyle.mediapipeDefault()));
  }

  @Test
  public void rejectsNullLayout() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            VisionOverlayRenderer.drawLandmarks(
                new Canvas(emptyBitmap()),
                Collections.emptyList(),
                Collections.emptySet(),
                null,
                OverlayStyle.mediapipeDefault()));
  }

  @Test
  public void rejectsNullStyle() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            VisionOverlayRenderer.drawLandmarks(
                new Canvas(emptyBitmap()),
                Collections.emptyList(),
                Collections.emptySet(),
                LAYOUT,
                null));
  }

  @Test
  public void mirroredLayout_drawsLandmarkOnTheRight() {
    OverlayLayout mirrored =
        OverlayLayout.create(SIZE, SIZE, SIZE, SIZE, RunningMode.IMAGE, 0, true);
    OverlayStyle style =
        OverlayStyle.builder().setLandmarkColor(Color.RED).setLandmarkRadius(10.f).build();
    Bitmap bitmap = emptyBitmap();
    // x=0.25 mirrors to 0.75 → pixel 75
    List<List<NormalizedLandmark>> landmarks =
        Collections.singletonList(
            Collections.singletonList(NormalizedLandmark.create(0.25f, 0.5f, 0.f)));

    VisionOverlayRenderer.drawLandmarks(
        new Canvas(bitmap), landmarks, Collections.emptySet(), mirrored, style);

    assertPixelCloseTo(bitmap, 75, 50, Color.RED);
  }

  private static Bitmap emptyBitmap() {
    Bitmap bitmap = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888);
    bitmap.eraseColor(Color.TRANSPARENT);
    return bitmap;
  }

  private static void assertPixelCloseTo(Bitmap bitmap, int x, int y, int expected) {
    int pixel = bitmap.getPixel(x, y);
    assertThat(Math.abs(Color.red(pixel) - Color.red(expected))).isLessThan(40);
    assertThat(Math.abs(Color.green(pixel) - Color.green(expected))).isLessThan(40);
    assertThat(Math.abs(Color.blue(pixel) - Color.blue(expected))).isLessThan(40);
    assertThat(Color.alpha(pixel)).isGreaterThan(32);
  }
}
