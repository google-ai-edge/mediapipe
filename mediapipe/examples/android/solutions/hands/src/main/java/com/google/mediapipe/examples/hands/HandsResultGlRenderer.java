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

package com.google.mediapipe.examples.hands;

import android.opengl.GLES20;
import android.opengl.Matrix;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutioncore.ResultGlBoundary;
import com.google.mediapipe.solutioncore.ResultGlRenderer;
import com.google.mediapipe.solutions.hands.Hands;
import com.google.mediapipe.solutions.hands.HandsResult;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

/** A custom implementation of {@link ResultGlRenderer} to render MediaPope Hands results. */
public class HandsResultGlRenderer implements ResultGlRenderer<HandsResult> {
  private static final String TAG = "HandsResultGlRenderer";

  private static final float CONNECTION_THICKNESS = 20.0f;
  private static final String VERTEX_SHADER =
      "uniform mat4 uTransformMatrix;\n"
          + "attribute vec4 vPosition;\n"
          + "void main() {\n"
          + "  gl_Position = uTransformMatrix * vPosition;\n"
          + "}";
  private static final String FRAGMENT_SHADER =
      "precision mediump float;\n"
          + "void main() {\n"
          + "  gl_FragColor = vec4(0, 1, 0, 1);\n"
          + "}";
  private int program;
  private int positionHandle;
  private int transformMatrixHandle;
  private final float[] transformMatrix = new float[16];
  private FloatBuffer vertexBuffer;

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
    transformMatrixHandle = GLES20.glGetUniformLocation(program, "uTransformMatrix");
  }

  @Override
  public void renderResult(HandsResult result, ResultGlBoundary boundary) {
    if (result == null) {
      return;
    }
    GLES20.glUseProgram(program);
    // Sets the transform matrix to align the result rendering with the scaled output texture.
    Matrix.setIdentityM(transformMatrix, 0);
    Matrix.scaleM(
        transformMatrix,
        0,
        2 / (boundary.right() - boundary.left()),
        2 / (boundary.top() - boundary.bottom()),
        1.0f);
    GLES20.glUniformMatrix4fv(transformMatrixHandle, 1, false, transformMatrix, 0);
    GLES20.glLineWidth(CONNECTION_THICKNESS);

    int numHands = result.multiHandLandmarks().size();
    for (int i = 0; i < numHands; ++i) {
      drawLandmarks(result.multiHandLandmarks().get(i).getLandmarkList());
    }
  }

  /**
   * Calls this to delete the shader program.
   *
   * <p>This is only necessary if one wants to release the program while keeping the context around.
   */
  public void release() {
    GLES20.glDeleteProgram(program);
  }

  // TODO: Better hand landmark and hand connection drawing.
  private void drawLandmarks(List<NormalizedLandmark> handLandmarkList) {
    for (Hands.Connection c : Hands.HAND_CONNECTIONS) {
      float[] vertex = new float[4];
      NormalizedLandmark start = handLandmarkList.get(c.start());
      vertex[0] = normalizedLandmarkValue(start.getX());
      vertex[1] = normalizedLandmarkValue(start.getY());
      NormalizedLandmark end = handLandmarkList.get(c.end());
      vertex[2] = normalizedLandmarkValue(end.getX());
      vertex[3] = normalizedLandmarkValue(end.getY());
      vertexBuffer =
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

  // Normalizes the value from the landmark value range:[0, 1] to the standard OpenGL coordinate
  // value range: [-1, 1].
  private float normalizedLandmarkValue(float value) {
    return value * 2 - 1;
  }
}
