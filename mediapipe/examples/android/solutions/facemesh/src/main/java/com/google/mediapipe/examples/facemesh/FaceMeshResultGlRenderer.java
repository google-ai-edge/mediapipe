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
import com.google.common.collect.ImmutableSet;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutioncore.ResultGlRenderer;
import com.google.mediapipe.solutions.facemesh.FaceMesh;
import com.google.mediapipe.solutions.facemesh.FaceMeshConnections;
import com.google.mediapipe.solutions.facemesh.FaceMeshResult;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

/** A custom implementation of {@link ResultGlRenderer} to render {@link FaceMeshResult}. */
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
      "uniform mat4 uProjectionMatrix;\n"
          + "attribute vec4 vPosition;\n"
          + "void main() {\n"
          + "  gl_Position = uProjectionMatrix * vPosition;\n"
          + "}";
  private static final String FRAGMENT_SHADER =
      "precision mediump float;\n"
          + "uniform vec4 uColor;\n"
          + "void main() {\n"
          + "  gl_FragColor = uColor;\n"
          + "}";
  private int program;
  private int positionHandle;
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
    projectionMatrixHandle = GLES20.glGetUniformLocation(program, "uProjectionMatrix");
    colorHandle = GLES20.glGetUniformLocation(program, "uColor");
  }

  @Override
  public void renderResult(FaceMeshResult result, float[] projectionMatrix) {
    if (result == null) {
      return;
    }
    GLES20.glUseProgram(program);
    GLES20.glUniformMatrix4fv(projectionMatrixHandle, 1, false, projectionMatrix, 0);

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
          FaceMeshConnections.FACEMESH_LEFT_EYEBROW,
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
      if (result.multiFaceLandmarks().get(i).getLandmarkCount()
          == FaceMesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES) {
        drawLandmarks(
            result.multiFaceLandmarks().get(i).getLandmarkList(),
            FaceMeshConnections.FACEMESH_RIGHT_IRIS,
            RIGHT_EYE_COLOR,
            RIGHT_EYE_THICKNESS);
        drawLandmarks(
            result.multiFaceLandmarks().get(i).getLandmarkList(),
            FaceMeshConnections.FACEMESH_LEFT_IRIS,
            LEFT_EYE_COLOR,
            LEFT_EYE_THICKNESS);
      }
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

  private void drawLandmarks(
      List<NormalizedLandmark> faceLandmarkList,
      ImmutableSet<FaceMeshConnections.Connection> connections,
      float[] colorArray,
      int thickness) {
    GLES20.glUniform4fv(colorHandle, 1, colorArray, 0);
    GLES20.glLineWidth(thickness);
    for (FaceMeshConnections.Connection c : connections) {
      NormalizedLandmark start = faceLandmarkList.get(c.start());
      NormalizedLandmark end = faceLandmarkList.get(c.end());
      float[] vertex = {start.getX(), start.getY(), end.getX(), end.getY()};
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
}
