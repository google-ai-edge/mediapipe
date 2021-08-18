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

import android.opengl.GLES20;
import android.opengl.Matrix;
import com.google.common.collect.ImmutableSet;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutioncore.ResultGlBoundary;
import com.google.mediapipe.solutioncore.ResultGlRenderer;
import com.google.mediapipe.solutions.facemesh.FaceMeshConnections;
import com.google.mediapipe.solutions.facemesh.FaceMeshResult;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

/** A custom implementation of {@link ResultGlRenderer} to render MediaPope FaceMesh results. */
public class FaceMeshResultGlRenderer implements ResultGlRenderer<FaceMeshResult> {
  private static final String TAG = "FaceMeshResultGlRenderer";

  private static final float[] TESSELATION_COLOR = new float[] {0.75f, 0.75f, 0.75f, 0.5f};
  private static final int TESSELATION_THICKNESS = 5;
  private static final float[] RIGHT_EYE_COLOR = new float[] {1f, 0.2f, 0.2f, 1f};
  private static final int RIGHT_EYE_THICKNESS = 8;
  private static final float[] RIGHT_EYEBROW_COLOR = new float[] {1f, 0.2f, 0.2f, 1f};
  private static final int RIGHT_EYEBROW_THICKNESS = 8;
  private static final float[] LEFT_EYE_COLOR = new float[] {0.2f, 1f, 0.2f, 1f};
  private static final int LEFT_EYE_THICKNESS = 8;
  private static final float[] LEFT_EYEBROW_COLOR = new float[] {0.2f, 1f, 0.2f, 1f};
  private static final int LEFT_EYEBROW_THICKNESS = 8;
  private static final float[] FACE_OVAL_COLOR = new float[] {0.9f, 0.9f, 0.9f, 1f};
  private static final int FACE_OVAL_THICKNESS = 8;
  private static final float[] LIPS_COLOR = new float[] {0.9f, 0.9f, 0.9f, 1f};
  private static final int LIPS_THICKNESS = 8;
  private static final String VERTEX_SHADER =
      "uniform mat4 uTransformMatrix;\n"
          + "attribute vec4 vPosition;\n"
          + "void main() {\n"
          + "  gl_Position = uTransformMatrix * vPosition;\n"
          + "}";
  private static final String FRAGMENT_SHADER =
      "precision mediump float;\n"
          + "uniform vec4 uColor;\n"
          + "void main() {\n"
          + "  gl_FragColor = uColor;\n"
          + "}";
  private int program;
  private int positionHandle;
  private int transformMatrixHandle;
  private int colorHandle;
  private final float[] transformMatrix = new float[16];

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
    colorHandle = GLES20.glGetUniformLocation(program, "uColor");
  }

  @Override
  public void renderResult(FaceMeshResult result, ResultGlBoundary boundary) {
    if (result == null) {
      return;
    }
    GLES20.glUseProgram(program);
    // Sets the transform matrix to align the result rendering with the scaled output texture.
    // Also flips the rendering vertically since OpenGL assumes the coordinate origin is at the
    // bottom-left corner, whereas MediaPipe landmark data assumes the coordinate origin is at the
    // top-left corner.
    Matrix.setIdentityM(transformMatrix, 0);
    Matrix.scaleM(
        transformMatrix,
        0,
        2 / (boundary.right() - boundary.left()),
        -2 / (boundary.top() - boundary.bottom()),
        1.0f);
    GLES20.glUniformMatrix4fv(transformMatrixHandle, 1, false, transformMatrix, 0);

    int numFaces = result.multiFaceLandmarks().size();
    for (int i = 0; i < numFaces; ++i) {
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_TESSELATION,
          TESSELATION_COLOR,
          TESSELATION_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_RIGHT_EYE,
          RIGHT_EYE_COLOR,
          RIGHT_EYE_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_RIGHT_EYEBROW,
          RIGHT_EYEBROW_COLOR,
          RIGHT_EYEBROW_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_LEFT_EYE,
          LEFT_EYE_COLOR,
          LEFT_EYE_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_LEFT_EYEBR0W,
          LEFT_EYEBROW_COLOR,
          LEFT_EYEBROW_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_FACE_OVAL,
          FACE_OVAL_COLOR,
          FACE_OVAL_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_LIPS,
          LIPS_COLOR,
          LIPS_THICKNESS);
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

  private void drawLandmarks(
      List<NormalizedLandmark> faceLandmarkList,
      ImmutableSet<FaceMeshConnections.Connection> connections,
      float[] colorArray,
      int thickness) {
    GLES20.glUniform4fv(colorHandle, 1, colorArray, 0);
    GLES20.glLineWidth(thickness);
    for (FaceMeshConnections.Connection c : connections) {
      float[] vertex = new float[4];
      NormalizedLandmark start = faceLandmarkList.get(c.start());
      vertex[0] = normalizedLandmarkValue(start.getX());
      vertex[1] = normalizedLandmarkValue(start.getY());
      NormalizedLandmark end = faceLandmarkList.get(c.end());
      vertex[2] = normalizedLandmarkValue(end.getX());
      vertex[3] = normalizedLandmarkValue(end.getY());
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

  // Normalizes the value from the landmark value range:[0, 1] to the standard OpenGL coordinate
  // value range: [-1, 1].
  private float normalizedLandmarkValue(float value) {
    return value * 2 - 1;
  }
}
