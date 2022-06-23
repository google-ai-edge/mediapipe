// Copyright 2021 The MediaPipe Authors.
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

package com.google.mediapipe.examples.facedetection;

import static java.lang.Math.min;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import androidx.appcompat.widget.AppCompatImageView;
import com.google.mediapipe.solutions.facedetection.FaceDetectionResult;
import com.google.mediapipe.solutions.facedetection.FaceKeypoint;
import com.google.mediapipe.formats.proto.DetectionProto.Detection;

/** An ImageView implementation for displaying {@link FaceDetectionResult}. */
public class FaceDetectionResultImageView extends AppCompatImageView {
  private static final String TAG = "FaceDetectionResultImageView";

  private static final int KEYPOINT_COLOR = Color.RED;
  private static final int KEYPOINT_RADIUS = 8; // Pixels
  private static final int BBOX_COLOR = Color.GREEN;
  private static final int BBOX_THICKNESS = 5; // Pixels
  private Bitmap latest;

  public FaceDetectionResultImageView(Context context) {
    super(context);
    setScaleType(AppCompatImageView.ScaleType.FIT_CENTER);
  }

  /**
   * Sets a {@link FaceDetectionResult} to render.
   *
   * @param result a {@link FaceDetectionResult} object that contains the solution outputs and the
   *     input {@link Bitmap}.
   */
  public void setFaceDetectionResult(FaceDetectionResult result) {
    if (result == null) {
      return;
    }
    Bitmap bmInput = result.inputBitmap();
    int width = bmInput.getWidth();
    int height = bmInput.getHeight();
    latest = Bitmap.createBitmap(width, height, bmInput.getConfig());
    Canvas canvas = new Canvas(latest);

    canvas.drawBitmap(bmInput, new Matrix(), null);
    int numDetectedFaces = result.multiFaceDetections().size();
    for (int i = 0; i < numDetectedFaces; ++i) {
      drawDetectionOnCanvas(result.multiFaceDetections().get(i), canvas, width, height);
    }
  }

  /** Updates the image view with the latest {@link FaceDetectionResult}. */
  public void update() {
    postInvalidate();
    if (latest != null) {
      setImageBitmap(latest);
    }
  }

  private void drawDetectionOnCanvas(Detection detection, Canvas canvas, int width, int height) {
    if (!detection.hasLocationData()) {
      return;
    }
    // Draw keypoints.
    Paint keypointPaint = new Paint();
    keypointPaint.setColor(KEYPOINT_COLOR);
    for (int i = 0; i < FaceKeypoint.NUM_KEY_POINTS; ++i) {
      int xPixel =
          min(
              (int) (detection.getLocationData().getRelativeKeypoints(i).getX() * width),
              width - 1);
      int yPixel =
          min(
              (int) (detection.getLocationData().getRelativeKeypoints(i).getY() * height),
              height - 1);
      canvas.drawCircle(xPixel, yPixel, KEYPOINT_RADIUS, keypointPaint);
    }
    if (!detection.getLocationData().hasRelativeBoundingBox()) {
      return;
    }
    // Draw bounding box.
    Paint bboxPaint = new Paint();
    bboxPaint.setColor(BBOX_COLOR);
    bboxPaint.setStyle(Paint.Style.STROKE);
    bboxPaint.setStrokeWidth(BBOX_THICKNESS);
    float left = detection.getLocationData().getRelativeBoundingBox().getXmin() * width;
    float top = detection.getLocationData().getRelativeBoundingBox().getYmin() * height;
    float right = left + detection.getLocationData().getRelativeBoundingBox().getWidth() * width;
    float bottom = top + detection.getLocationData().getRelativeBoundingBox().getHeight() * height;
    canvas.drawRect(left, top, right, bottom, bboxPaint);
  }
}
