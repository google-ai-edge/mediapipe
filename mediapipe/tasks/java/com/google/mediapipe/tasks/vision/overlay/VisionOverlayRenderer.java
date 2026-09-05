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

import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Paint.Align;
import android.graphics.Paint.Style;
import android.graphics.Rect;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.Connection;
import com.google.mediapipe.tasks.components.containers.Detection;
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizerResult;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Draws MediaPipe Tasks vision results onto an {@link Canvas}.
 *
 * <p>Use from Jetpack Compose:
 *
 * <pre>{@code
 * Canvas(Modifier.fillMaxSize()) {
 *   val layout = OverlayLayout.create(
 *       size.width.toInt(), size.height.toInt(),
 *       imageWidth, imageHeight, RunningMode.LIVE_STREAM,
 *       rotationDegrees = 0, mirrored = lensFacing == LENS_FACING_FRONT)
 *   drawIntoCanvas { native ->
 *     VisionOverlayRenderer.drawHands(
 *         native.nativeCanvas, result, layout, OverlayStyle.mediapipeDefault(density.density))
 *   }
 * }
 * }</pre>
 *
 * <p>Null or empty results are no-ops. Connection indices outside a landmark list are skipped.
 */
public final class VisionOverlayRenderer {
  private VisionOverlayRenderer() {}

  /** Draws {@link HandLandmarker} skeletons. */
  public static void drawHands(
      Canvas canvas, HandLandmarkerResult result, OverlayLayout layout, OverlayStyle style) {
    if (result == null) {
      return;
    }
    drawLandmarks(canvas, result.landmarks(), HandLandmarker.HAND_CONNECTIONS, layout, style);
  }

  /** Draws {@link PoseLandmarker} skeletons. */
  public static void drawPose(
      Canvas canvas, PoseLandmarkerResult result, OverlayLayout layout, OverlayStyle style) {
    if (result == null) {
      return;
    }
    drawLandmarks(canvas, result.landmarks(), PoseLandmarker.POSE_LANDMARKS, layout, style);
  }

  /** Draws {@link FaceLandmarker} contours (connectors, not the full tesselation). */
  public static void drawFaceLandmarks(
      Canvas canvas, FaceLandmarkerResult result, OverlayLayout layout, OverlayStyle style) {
    if (result == null) {
      return;
    }
    drawLandmarks(
        canvas, result.faceLandmarks(), FaceLandmarker.FACE_LANDMARKS_CONNECTORS, layout, style);
  }

  /** Draws {@link ObjectDetector} boxes and labels. */
  public static void drawObjects(
      Canvas canvas, ObjectDetectorResult result, OverlayLayout layout, OverlayStyle style) {
    if (result == null) {
      return;
    }
    drawDetections(canvas, result.detections(), layout, style, /* drawKeypoints= */ false);
  }

  /** Draws {@link FaceDetector} boxes, labels, and keypoints. */
  public static void drawFaceDetections(
      Canvas canvas, FaceDetectorResult result, OverlayLayout layout, OverlayStyle style) {
    if (result == null) {
      return;
    }
    drawDetections(canvas, result.detections(), layout, style, /* drawKeypoints= */ true);
  }

  /**
   * Draws {@link GestureRecognizer} hand skeletons and the top gesture label near the wrist
   * (landmark 0).
   */
  public static void drawGestures(
      Canvas canvas, GestureRecognizerResult result, OverlayLayout layout, OverlayStyle style) {
    if (result == null) {
      return;
    }
    drawLandmarks(canvas, result.landmarks(), HandLandmarker.HAND_CONNECTIONS, layout, style);
    List<List<NormalizedLandmark>> hands = result.landmarks();
    List<List<Category>> gestures = result.gestures();
    int count = Math.min(hands.size(), gestures.size());
    for (int i = 0; i < count; i++) {
      List<NormalizedLandmark> hand = hands.get(i);
      List<Category> categories = gestures.get(i);
      if (hand.isEmpty() || categories.isEmpty()) {
        continue;
      }
      drawLabel(
          canvas,
          layout.mapX(hand.get(0).x()),
          layout.mapY(hand.get(0).y()),
          formatCategory(categories.get(0)),
          style);
    }
  }

  /** Draws {@link HolisticLandmarker} face, pose, and both hands. */
  public static void drawHolistic(
      Canvas canvas, HolisticLandmarkerResult result, OverlayLayout layout, OverlayStyle style) {
    if (result == null) {
      return;
    }
    if (!result.faceLandmarks().isEmpty()) {
      drawLandmarks(
          canvas,
          Collections.singletonList(result.faceLandmarks()),
          FaceLandmarker.FACE_LANDMARKS_CONNECTORS,
          layout,
          style);
    }
    if (!result.poseLandmarks().isEmpty()) {
      drawLandmarks(
          canvas,
          Collections.singletonList(result.poseLandmarks()),
          PoseLandmarker.POSE_LANDMARKS,
          layout,
          style);
    }
    if (!result.leftHandLandmarks().isEmpty()) {
      drawLandmarks(
          canvas,
          Collections.singletonList(result.leftHandLandmarks()),
          HandLandmarker.HAND_CONNECTIONS,
          layout,
          style);
    }
    if (!result.rightHandLandmarks().isEmpty()) {
      drawLandmarks(
          canvas,
          Collections.singletonList(result.rightHandLandmarks()),
          HandLandmarker.HAND_CONNECTIONS,
          layout,
          style);
    }
  }

  /**
   * Draws one or more landmark lists and the given connections. Safe to call from Compose or a
   * custom View.
   */
  public static void drawLandmarks(
      Canvas canvas,
      List<List<NormalizedLandmark>> multiLandmarks,
      Set<Connection> connections,
      OverlayLayout layout,
      OverlayStyle style) {
    requireDrawArgs(canvas, layout, style);
    if (multiLandmarks == null || multiLandmarks.isEmpty()) {
      return;
    }
    Paint linePaint = connectionPaint(style);
    Paint pointPaint = landmarkPaint(style);
    for (List<NormalizedLandmark> landmarks : multiLandmarks) {
      if (landmarks == null || landmarks.isEmpty()) {
        continue;
      }
      if (connections != null) {
        for (Connection connection : connections) {
          if (connection == null) {
            continue;
          }
          int start = connection.start();
          int end = connection.end();
          if (start < 0 || end < 0 || start >= landmarks.size() || end >= landmarks.size()) {
            continue;
          }
          NormalizedLandmark from = landmarks.get(start);
          NormalizedLandmark to = landmarks.get(end);
          canvas.drawLine(
              layout.mapX(from.x()),
              layout.mapY(from.y()),
              layout.mapX(to.x()),
              layout.mapY(to.y()),
              linePaint);
        }
      }
      for (NormalizedLandmark landmark : landmarks) {
        canvas.drawCircle(
            layout.mapX(landmark.x()),
            layout.mapY(landmark.y()),
            style.landmarkRadius(),
            pointPaint);
      }
    }
  }

  /**
   * Draws detection bounding boxes (image pixel space) and the top category label.
   *
   * @param drawKeypoints when {@code true}, also draws {@link Detection#keypoints()}
   */
  public static void drawDetections(
      Canvas canvas,
      List<Detection> detections,
      OverlayLayout layout,
      OverlayStyle style,
      boolean drawKeypoints) {
    requireDrawArgs(canvas, layout, style);
    if (detections == null || detections.isEmpty()) {
      return;
    }
    Paint boxPaint = boxPaint(style);
    Paint keypointPaint = landmarkPaint(style);
    for (Detection detection : detections) {
      if (detection == null) {
        continue;
      }
      RectF box = layout.mapImageRect(detection.boundingBox());
      canvas.drawRect(box, boxPaint);
      if (!detection.categories().isEmpty()) {
        drawLabel(canvas, box.left, box.top, formatCategory(detection.categories().get(0)), style);
      }
      if (drawKeypoints && detection.keypoints().isPresent()) {
        for (NormalizedKeypoint keypoint : detection.keypoints().get()) {
          canvas.drawCircle(
              layout.mapX(keypoint.x()),
              layout.mapY(keypoint.y()),
              style.landmarkRadius(),
              keypointPaint);
        }
      }
    }
  }

  static String formatCategory(Category category) {
    String name = category.displayName();
    if (name == null || name.isEmpty()) {
      name = category.categoryName();
    }
    if (name == null) {
      name = "";
    }
    return String.format(Locale.US, "%s %.2f", name, category.score());
  }

  private static void drawLabel(
      Canvas canvas, float left, float top, String text, OverlayStyle style) {
    Paint textPaint = textPaint(style);
    Paint backgroundPaint = textBackgroundPaint(style);
    Rect bounds = new Rect();
    textPaint.getTextBounds(text, 0, text.length(), bounds);
    float padding = style.textPadding();
    float bgTop = top;
    float bgBottom = top + bounds.height() + 2.f * padding;
    float bgRight = left + bounds.width() + 2.f * padding;
    canvas.drawRect(left, bgTop, bgRight, bgBottom, backgroundPaint);
    // getTextBounds origin is the text baseline; pad then offset by -bounds.top.
    canvas.drawText(text, left + padding, top + padding - bounds.top, textPaint);
  }

  private static void requireDrawArgs(Canvas canvas, OverlayLayout layout, OverlayStyle style) {
    if (canvas == null) {
      throw new IllegalArgumentException("Canvas must not be null");
    }
    if (layout == null) {
      throw new IllegalArgumentException("OverlayLayout must not be null");
    }
    if (style == null) {
      throw new IllegalArgumentException("OverlayStyle must not be null");
    }
  }

  private static Paint landmarkPaint(OverlayStyle style) {
    Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    paint.setColor(style.landmarkColor());
    paint.setStyle(Style.FILL);
    return paint;
  }

  private static Paint connectionPaint(OverlayStyle style) {
    Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    paint.setColor(style.connectionColor());
    paint.setStyle(Style.STROKE);
    paint.setStrokeWidth(style.connectionStrokeWidth());
    paint.setStrokeCap(Paint.Cap.ROUND);
    return paint;
  }

  private static Paint boxPaint(OverlayStyle style) {
    Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    paint.setColor(style.boxColor());
    paint.setStyle(Style.STROKE);
    paint.setStrokeWidth(style.boxStrokeWidth());
    return paint;
  }

  private static Paint textPaint(OverlayStyle style) {
    Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    paint.setColor(style.textColor());
    paint.setStyle(Style.FILL);
    paint.setTextSize(style.textSize());
    paint.setTextAlign(Align.LEFT);
    return paint;
  }

  private static Paint textBackgroundPaint(OverlayStyle style) {
    Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    paint.setColor(style.textBackgroundColor());
    paint.setStyle(Style.FILL);
    return paint;
  }
}
