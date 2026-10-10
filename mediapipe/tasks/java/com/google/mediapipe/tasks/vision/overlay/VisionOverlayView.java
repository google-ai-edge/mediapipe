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

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.util.AttributeSet;
import android.view.View;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult;
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizerResult;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;

/**
 * Transparent overlay {@link View} that draws MediaPipe Tasks vision results.
 *
 * <p>This is the Compose integration path that does not require a Compose dependency in
 * {@code tasks-vision}: wrap the view with {@code AndroidView}, stacked on {@code PreviewView}.
 *
 * <pre>{@code
 * Box(Modifier.fillMaxSize()) {
 *   AndroidView(factory = { PreviewView(it) }, modifier = Modifier.fillMaxSize())
 *   AndroidView(
 *       factory = { VisionOverlayView(it) },
 *       modifier = Modifier.fillMaxSize(),
 *       update = { overlay ->
 *         overlay.setMirrored(lensFacing == CameraSelector.LENS_FACING_FRONT)
 *         overlay.setHandLandmarkerResult(
 *             result, imageWidth, imageHeight, RunningMode.LIVE_STREAM)
 *       })
 * }
 * }</pre>
 *
 * <p>{@link #setHandLandmarkerResult} and the other setters are safe to call from the MediaPipe
 * live-stream callback thread; they {@link #postInvalidate()}.
 */
public final class VisionOverlayView extends View {
  private enum Kind {
    NONE,
    HANDS,
    POSE,
    FACE_LANDMARKS,
    OBJECTS,
    FACE_DETECTIONS,
    GESTURES,
    HOLISTIC
  }

  private Kind kind = Kind.NONE;
  private HandLandmarkerResult hands;
  private PoseLandmarkerResult pose;
  private FaceLandmarkerResult faceLandmarks;
  private ObjectDetectorResult objects;
  private FaceDetectorResult faceDetections;
  private GestureRecognizerResult gestures;
  private HolisticLandmarkerResult holistic;
  private int imageWidth = 1;
  private int imageHeight = 1;
  private RunningMode runningMode = RunningMode.IMAGE;
  private int rotationDegrees = 0;
  private boolean mirrored = false;
  private OverlayStyle style;

  public VisionOverlayView(Context context) {
    this(context, null);
  }

  public VisionOverlayView(Context context, AttributeSet attrs) {
    this(context, attrs, 0);
  }

  public VisionOverlayView(Context context, AttributeSet attrs, int defStyleAttr) {
    super(context, attrs, defStyleAttr);
    setWillNotDraw(false);
    setBackgroundColor(Color.TRANSPARENT);
    style = OverlayStyle.mediapipeDefault(context.getResources().getDisplayMetrics().density);
  }

  /** Replaces the default MediaPipe colors / sizes. */
  public void setStyle(OverlayStyle style) {
    if (style == null) {
      throw new IllegalArgumentException("OverlayStyle must not be null");
    }
    this.style = style;
    postInvalidate();
  }

  public OverlayStyle getStyle() {
    return style;
  }

  /**
   * Flip overlay X to match a mirrored front-camera preview. Default {@code false}.
   *
   * <p>Call this when switching cameras; it applies to the next draw of whatever result is set.
   */
  public void setMirrored(boolean mirrored) {
    this.mirrored = mirrored;
    postInvalidate();
  }

  public boolean isMirrored() {
    return mirrored;
  }

  /**
   * Clockwise camera-frame rotation in degrees (0/90/180/270). Used for object/face boxes in
   * sensor orientation. Default {@code 0}.
   */
  public void setRotationDegrees(int rotationDegrees) {
    this.rotationDegrees = OverlayLayout.normalizeRotation(rotationDegrees);
    postInvalidate();
  }

  public int getRotationDegrees() {
    return rotationDegrees;
  }

  /** Clears the current result and redraws an empty overlay. */
  public void clear() {
    kind = Kind.NONE;
    hands = null;
    pose = null;
    faceLandmarks = null;
    objects = null;
    faceDetections = null;
    gestures = null;
    holistic = null;
    postInvalidate();
  }

  public void setHandLandmarkerResult(
      HandLandmarkerResult result, int imageWidth, int imageHeight, RunningMode runningMode) {
    setImage(imageWidth, imageHeight, runningMode);
    kind = Kind.HANDS;
    hands = result;
    postInvalidate();
  }

  public void setPoseLandmarkerResult(
      PoseLandmarkerResult result, int imageWidth, int imageHeight, RunningMode runningMode) {
    setImage(imageWidth, imageHeight, runningMode);
    kind = Kind.POSE;
    pose = result;
    postInvalidate();
  }

  public void setFaceLandmarkerResult(
      FaceLandmarkerResult result, int imageWidth, int imageHeight, RunningMode runningMode) {
    setImage(imageWidth, imageHeight, runningMode);
    kind = Kind.FACE_LANDMARKS;
    faceLandmarks = result;
    postInvalidate();
  }

  public void setObjectDetectorResult(
      ObjectDetectorResult result, int imageWidth, int imageHeight, RunningMode runningMode) {
    setImage(imageWidth, imageHeight, runningMode);
    kind = Kind.OBJECTS;
    objects = result;
    postInvalidate();
  }

  public void setFaceDetectorResult(
      FaceDetectorResult result, int imageWidth, int imageHeight, RunningMode runningMode) {
    setImage(imageWidth, imageHeight, runningMode);
    kind = Kind.FACE_DETECTIONS;
    faceDetections = result;
    postInvalidate();
  }

  public void setGestureRecognizerResult(
      GestureRecognizerResult result, int imageWidth, int imageHeight, RunningMode runningMode) {
    setImage(imageWidth, imageHeight, runningMode);
    kind = Kind.GESTURES;
    gestures = result;
    postInvalidate();
  }

  public void setHolisticLandmarkerResult(
      HolisticLandmarkerResult result, int imageWidth, int imageHeight, RunningMode runningMode) {
    setImage(imageWidth, imageHeight, runningMode);
    kind = Kind.HOLISTIC;
    holistic = result;
    postInvalidate();
  }

  @Override
  protected void onDraw(Canvas canvas) {
    super.onDraw(canvas);
    int contentWidth = getWidth() - getPaddingLeft() - getPaddingRight();
    int contentHeight = getHeight() - getPaddingTop() - getPaddingBottom();
    if (kind == Kind.NONE || contentWidth <= 0 || contentHeight <= 0) {
      return;
    }
    OverlayLayout layout =
        OverlayLayout.create(
            contentWidth,
            contentHeight,
            imageWidth,
            imageHeight,
            runningMode,
            rotationDegrees,
            mirrored);
    canvas.save();
    canvas.translate(getPaddingLeft(), getPaddingTop());
    switch (kind) {
      case HANDS:
        VisionOverlayRenderer.drawHands(canvas, hands, layout, style);
        break;
      case POSE:
        VisionOverlayRenderer.drawPose(canvas, pose, layout, style);
        break;
      case FACE_LANDMARKS:
        VisionOverlayRenderer.drawFaceLandmarks(canvas, faceLandmarks, layout, style);
        break;
      case OBJECTS:
        VisionOverlayRenderer.drawObjects(canvas, objects, layout, style);
        break;
      case FACE_DETECTIONS:
        VisionOverlayRenderer.drawFaceDetections(canvas, faceDetections, layout, style);
        break;
      case GESTURES:
        VisionOverlayRenderer.drawGestures(canvas, gestures, layout, style);
        break;
      case HOLISTIC:
        VisionOverlayRenderer.drawHolistic(canvas, holistic, layout, style);
        break;
      case NONE:
        break;
    }
    canvas.restore();
  }

  private void setImage(int imageWidth, int imageHeight, RunningMode runningMode) {
    if (imageWidth <= 0 || imageHeight <= 0) {
      throw new IllegalArgumentException(
          "Image size must be positive, found: " + imageWidth + "x" + imageHeight);
    }
    if (runningMode == null) {
      throw new IllegalArgumentException("RunningMode must not be null");
    }
    this.imageWidth = imageWidth;
    this.imageHeight = imageHeight;
    this.runningMode = runningMode;
  }
}
