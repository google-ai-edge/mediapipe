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

import android.opengl.GLES20;
import com.google.mediapipe.solutioncore.ResultGlRenderer;
import com.google.mediapipe.solutions.facedetection.FaceDetectionResult;
import com.google.mediapipe.solutions.facedetection.FaceKeypoint;
import com.google.mediapipe.formats.proto.DetectionProto.Detection;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/** A custom implementation of {@link ResultGlRenderer} to render {@link FaceDetectionResult}. */
public class FaceDetectionResultGlRenderer implements ResultGlRenderer<FaceDetectionResult> {
  private static final String TAG = "FaceDetectionResultGlRenderer";

  private static final float[] KEYPOINT_COLOR = new float[] {1f, 0f, 0f, 1f};
  private static final float KEYPOINT_SIZE = 16f;
  private static final float[] BBOX_COLOR = new float[] {0f, 1f, 0f, 1f};
  private static final int BBOX_THICKNESS = 8;
  private static final String VERTEX_SHADER =
      "uniform mat4 uProjectionMatrix;\n"
          + "uniform float uPointSize;\n"
          + "attribute vec4 vPosition;\n"
          + "void main() {\n"
          + "  gl_Position = uProjectionMatrix * vPosition;\n"
          + "  gl_PointSize = uPointSize;"
          + "}";
  private static final String FRAGMENT_SHADER =
      "precision mediump float;\n"
          + "uniform vec4 uColor;\n"
          + "void main() {\n"
          + "  gl_FragColor = uColor;\n"
          + "}";
  private int program;
  private int positionHandle;
  private int pointSizeHandle;
  private int projectionMatrixHandle;
  private int colorHandle;

  private int loadShader(int type, String shaderCode) {
    int shader = GLES20.glCreateShader(type);
    GLES20.glShaderSource(shader, shaderCode);
    GLES20.glCompileShader(shader);
    return shader;
  }

  @Override
  public void setupRendering() {
    program = GLES20.glCreateProgram();
    int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER);
    int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
    GLES20.glAttachShader(program, vertexShader);
    GLES20.glAttachShader(program, fragmentShader);
    GLES20.glLinkProgram(program);
    positionHandle = GLES20.glGetAttribLocation(program, "vPosition");
    pointSizeHandle = GLES20.glGetUniformLocation(program, "uPointSize");
    projectionMatrixHandle = GLES20.glGetUniformLocation(program, "uProjectionMatrix");
    colorHandle = GLES20.glGetUniformLocation(program, "uColor");
  }

  @Override
  public void renderResult(FaceDetectionResult result, float[] projectionMatrix) {
    if (result == null) {
      return;
    }
    GLES20.glUseProgram(program);
    GLES20.glUniformMatrix4fv(projectionMatrixHandle, 1, false, projectionMatrix, 0);
    GLES20.glUniform1f(pointSizeHandle, KEYPOINT_SIZE);
    int numDetectedFaces = result.multiFaceDetections().size();
    for (int i = 0; i < numDetectedFaces; ++i) {
      drawDetection(result.multiFaceDetections().get(i));
    }
  }

  /**
   * Deletes the shader program.
   *
   * <p>This is only necessary if one wants to release the program while keeping the context around.
   */
  public void release() {
    GLES20.glDeleteProgram(program);
  }

  private void drawDetection(Detection detection) {
    if (!detection.hasLocationData()) {
      return;
    }
    // Draw keypoints.
    float[] points = new float[FaceKeypoint.NUM_KEY_POINTS * 2];
    for (int i = 0; i < FaceKeypoint.NUM_KEY_POINTS; ++i) {
      points[2 * i] = detection.getLocationData().getRelativeKeypoints(i).getX();
      points[2 * i + 1] = detection.getLocationData().getRelativeKeypoints(i).getY();
    }
    GLES20.glUniform4fv(colorHandle, 1, KEYPOINT_COLOR, 0);
    FloatBuffer vertexBuffer =
        ByteBuffer.allocateDirect(points.length * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(points);
    vertexBuffer.position(0);
    GLES20.glEnableVertexAttribArray(positionHandle);
    GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);
    GLES20.glDrawArrays(GLES20.GL_POINTS, 0, FaceKeypoint.NUM_KEY_POINTS);
    if (!detection.getLocationData().hasRelativeBoundingBox()) {
      return;
    }
    // Draw bounding box.
    float left = detection.getLocationData().getRelativeBoundingBox().getXmin();
    float top = detection.getLocationData().getRelativeBoundingBox().getYmin();
    float right = left + detection.getLocationData().getRelativeBoundingBox().getWidth();
    float bottom = top + detection.getLocationData().getRelativeBoundingBox().getHeight();
    drawLine(top, left, top, right);
    drawLine(bottom, left, bottom, right);
    drawLine(top, left, bottom, left);
    drawLine(top, right, bottom, right);
  }

  private void drawLine(float y1, float x1, float y2, float x2) {
    GLES20.glUniform4fv(colorHandle, 1, BBOX_COLOR, 0);
    GLES20.glLineWidth(BBOX_THICKNESS);
    float[] vertex = {x1, y1, x2, y2};
    FloatBuffer vertexBuffer =
        ByteBuffer.allocateDirect(vertex.length * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(vertex);
    vertexBuffer.position(0);
    GLES20.glEnableVertexAttribArray(positionHandle);
    GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);
    GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2);
  }
}
