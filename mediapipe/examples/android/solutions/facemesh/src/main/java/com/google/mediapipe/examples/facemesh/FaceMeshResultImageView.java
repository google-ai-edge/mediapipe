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

package com.google.mediapipe.examples.facemesh;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import androidx.appcompat.widget.AppCompatImageView;
import android.util.Size;
import com.google.common.collect.ImmutableSet;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutions.facemesh.FaceMesh;
import com.google.mediapipe.solutions.facemesh.FaceMeshConnections;
import com.google.mediapipe.solutions.facemesh.FaceMeshResult;
import java.util.List;

/** An ImageView implementation for displaying {@link FaceMeshResult}. */
public class FaceMeshResultImageView extends AppCompatImageView {
  private static final String TAG = "FaceMeshResultImageView";

  private static final int TESSELATION_COLOR = Color.parseColor("#70C0C0C0");
  private static final int TESSELATION_THICKNESS = 3; // Pixels
  private static final int RIGHT_EYE_COLOR = Color.parseColor("#FF3030");
  private static final int RIGHT_EYE_THICKNESS = 5; // Pixels
  private static final int RIGHT_EYEBROW_COLOR = Color.parseColor("#FF3030");
  private static final int RIGHT_EYEBROW_THICKNESS = 5; // Pixels
  private static final int LEFT_EYE_COLOR = Color.parseColor("#30FF30");
  private static final int LEFT_EYE_THICKNESS = 5; // Pixels
  private static final int LEFT_EYEBROW_COLOR = Color.parseColor("#30FF30");
  private static final int LEFT_EYEBROW_THICKNESS = 5; // Pixels
  private static final int FACE_OVAL_COLOR = Color.parseColor("#E0E0E0");
  private static final int FACE_OVAL_THICKNESS = 5; // Pixels
  private static final int LIPS_COLOR = Color.parseColor("#E0E0E0");
  private static final int LIPS_THICKNESS = 5; // Pixels
  private Bitmap latest;

  public FaceMeshResultImageView(Context context) {
    super(context);
    setScaleType(AppCompatImageView.ScaleType.FIT_CENTER);
  }

  /**
   * Sets a {@link FaceMeshResult} to render.
   *
   * @param result a {@link FaceMeshResult} object that contains the solution outputs and the input
   *     {@link Bitmap}.
   */
  public void setFaceMeshResult(FaceMeshResult result) {
    if (result == null) {
      return;
    }
    Bitmap bmInput = result.inputBitmap();
    int width = bmInput.getWidth();
    int height = bmInput.getHeight();
    latest = Bitmap.createBitmap(width, height, bmInput.getConfig());
    Canvas canvas = new Canvas(latest);
    Size imageSize = new Size(width, height);
    canvas.drawBitmap(bmInput, new Matrix(), null);
    int numFaces = result.multiFaceLandmarks().size();
    for (int i = 0; i < numFaces; ++i) {
      drawLandmarksOnCanvas(
          canvas,
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_TESSELATION,
          imageSize,
          TESSELATION_COLOR,
          TESSELATION_THICKNESS);
      drawLandmarksOnCanvas(
          canvas,
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_RIGHT_EYE,
          imageSize,
          RIGHT_EYE_COLOR,
          RIGHT_EYE_THICKNESS);
      drawLandmarksOnCanvas(
          canvas,
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_RIGHT_EYEBROW,
          imageSize,
          RIGHT_EYEBROW_COLOR,
          RIGHT_EYEBROW_THICKNESS);
      drawLandmarksOnCanvas(
          canvas,
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_LEFT_EYE,
          imageSize,
          LEFT_EYE_COLOR,
          LEFT_EYE_THICKNESS);
      drawLandmarksOnCanvas(
          canvas,
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_LEFT_EYEBROW,
          imageSize,
          LEFT_EYEBROW_COLOR,
          LEFT_EYEBROW_THICKNESS);
      drawLandmarksOnCanvas(
          canvas,
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_FACE_OVAL,
          imageSize,
          FACE_OVAL_COLOR,
          FACE_OVAL_THICKNESS);
      drawLandmarksOnCanvas(
          canvas,
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_LIPS,
          imageSize,
          LIPS_COLOR,
          LIPS_THICKNESS);
      if (result.multiFaceLandmarks().get(i).getLandmarkCount()
          == FaceMesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES) {
        drawLandmarksOnCanvas(
            canvas,
            result.multiFaceLandmarks().get(i).getLandmarkList(),
            FaceMeshConnections.FACEMESH_RIGHT_IRIS,
            imageSize,
            RIGHT_EYE_COLOR,
            RIGHT_EYE_THICKNESS);
        drawLandmarksOnCanvas(
            canvas,
            result.multiFaceLandmarks().get(i).getLandmarkList(),
            FaceMeshConnections.FACEMESH_LEFT_IRIS,
            imageSize,
            LEFT_EYE_COLOR,
            LEFT_EYE_THICKNESS);
      }
    }
  }

  /** Updates the image view with the latest {@link FaceMeshResult}. */
  public void update() {
    postInvalidate();
    if (latest != null) {
      setImageBitmap(latest);
    }
  }

  private void drawLandmarksOnCanvas(
      Canvas canvas,
      List<NormalizedLandmark> faceLandmarkList,
      ImmutableSet<FaceMeshConnections.Connection> connections,
      Size imageSize,
      int color,
      int thickness) {
    // Draw connections.
    for (FaceMeshConnections.Connection c : connections) {
      Paint connectionPaint = new Paint();
      connectionPaint.setColor(color);
      connectionPaint.setStrokeWidth(thickness);
      NormalizedLandmark start = faceLandmarkList.get(c.start());
      NormalizedLandmark end = faceLandmarkList.get(c.end());
      canvas.drawLine(
          start.getX() * imageSize.getWidth(),
          start.getY() * imageSize.getHeight(),
          end.getX() * imageSize.getWidth(),
          end.getY() * imageSize.getHeight(),
          connectionPaint);
    }
  }
}
