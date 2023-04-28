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

package com.google.mediapipe.tasks.examples.objectdetector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import androidx.appcompat.widget.AppCompatImageView;
import com.google.mediapipe.framework.image.BitmapExtractor;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Detection;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectionResult;

/** An ImageView implementation for displaying {@link ObjectDetectionResult}. */
public class ObjectDetectionResultImageView extends AppCompatImageView {
  private static final String TAG = "ObjectDetectionResultImageView";

  private static final int BBOX_COLOR = Color.GREEN;
  private static final int BBOX_THICKNESS = 5; // Pixels
  private Bitmap latest;

  public ObjectDetectionResultImageView(Context context) {
    super(context);
    setScaleType(AppCompatImageView.ScaleType.FIT_CENTER);
  }

  /**
   * Sets a {@link MPImage} and an {@link ObjectDetectionResult} to render.
   *
   * @param image a {@link MPImage} object for annotation.
   * @param result an {@link ObjectDetectionResult} object that contains the detection result.
   */
  public void setData(MPImage image, ObjectDetectionResult result) {
    if (image == null || result == null) {
      return;
    }
    latest = BitmapExtractor.extract(image);
    Canvas canvas = new Canvas(latest);
    canvas.drawBitmap(latest, new Matrix(), null);
    for (int i = 0; i < result.detections().size(); ++i) {
      drawDetectionOnCanvas(result.detections().get(i), canvas);
    }
  }

  /** Updates the image view with the latest {@link ObjectDetectionResult}. */
  public void update() {
    postInvalidate();
    if (latest != null) {
      setImageBitmap(latest);
    }
  }

  private void drawDetectionOnCanvas(Detection detection, Canvas canvas) {
    // TODO: Draws the category and the score per bounding box.
    // Draws bounding box.
    Paint bboxPaint = new Paint();
    bboxPaint.setColor(BBOX_COLOR);
    bboxPaint.setStyle(Paint.Style.STROKE);
    bboxPaint.setStrokeWidth(BBOX_THICKNESS);
    canvas.drawRect(detection.boundingBox(), bboxPaint);
  }
}
