// Copyright 2019 The MediaPipe Authors.
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

package com.google.mediapipe.glutil;

import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.view.Surface;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * Textures from {@link SurfaceTexture} are only supposed to be bound to target {@link
 * GLES11Ext#GL_TEXTURE_EXTERNAL_OES}, which is accessed using samplerExternalOES in the shader.
 * This means they cannot be used with a regular shader that expects a sampler2D. This class renders
 * the external texture to the current framebuffer. By binding the framebuffer to a texture, this
 * can be used to convert the input into a normal 2D texture.
 */
public class ExternalTextureRenderer {
  private static final FloatBuffer TEXTURE_VERTICES =
      ShaderUtil.floatBuffer(
          0.0f, 0.0f, // bottom left
          1.0f, 0.0f, // bottom right
          0.0f, 1.0f, // top left
          1.0f, 1.0f // top right
          );

  private static final FloatBuffer FLIPPED_TEXTURE_VERTICES =
      ShaderUtil.floatBuffer(
          0.0f, 1.0f, // top left
          1.0f, 1.0f, // top right
          0.0f, 0.0f, // bottom left
          1.0f, 0.0f // bottom right
          );
  private static final Vertex BOTTOM_LEFT = new Vertex(-1.0f, -1.0f);
  private static final Vertex BOTTOM_RIGHT = new Vertex(1.0f, -1.0f);
  private static final Vertex TOP_LEFT = new Vertex(-1.0f, 1.0f);
  private static final Vertex TOP_RIGHT = new Vertex(1.0f, 1.0f);
  private static final Vertex[] POSITION_VERTICIES = {
    BOTTOM_LEFT, BOTTOM_RIGHT, TOP_LEFT, TOP_RIGHT
  };
  private static final FloatBuffer POSITION_VERTICIES_0 = fb(POSITION_VERTICIES, 0, 1, 2, 3);
  private static final FloatBuffer POSITION_VERTICIES_90 = fb(POSITION_VERTICIES, 2, 0, 3, 1);
  private static final FloatBuffer POSITION_VERTICIES_180 = fb(POSITION_VERTICIES, 3, 2, 1, 0);
  private static final FloatBuffer POSITION_VERTICIES_270 = fb(POSITION_VERTICIES, 1, 3, 0, 2);

  private static final String TAG = "ExternalTextureRend"; // Max length of a tag is 23.
  private static final int ATTRIB_POSITION = 1;
  private static final int ATTRIB_TEXTURE_COORDINATE = 2;

  private int program = 0;
  private int frameUniform;
  private int textureTransformUniform;
  private float[] textureTransformMatrix = new float[16];
  private boolean flipY;
  private int rotation = Surface.ROTATION_0;
  private boolean doExplicitCpuSync = true;

  /** Call this to setup the shader program before rendering. */
  public void setup() {
    Map<String, Integer> attributeLocations = new HashMap<>();
    attributeLocations.put("position", ATTRIB_POSITION);
    attributeLocations.put("texture_coordinate", ATTRIB_TEXTURE_COORDINATE);
    program =
        ShaderUtil.createProgram(
            CommonShaders.VERTEX_SHADER,
            CommonShaders.FRAGMENT_SHADER_EXTERNAL,
            attributeLocations);
    frameUniform = GLES20.glGetUniformLocation(program, "video_frame");
    textureTransformUniform = GLES20.glGetUniformLocation(program, "texture_transform");
    ShaderUtil.checkGlError("glGetUniformLocation");
  }

  /**
   * Flips rendering output vertically, useful for conversion between coordinate systems with
   * top-left v.s. bottom-left origins. Effective in subsequent {@link #render(SurfaceTexture)}
   * calls.
   */
  public void setFlipY(boolean flip) {
    flipY = flip;
  }

  /**
   * Rotates the rendering output, useful for supporting landscape orientations. The value should
   * correspond to Display.getRotation(), e.g. Surface.ROTATION_0. Flipping (if any) is applied
   * before rotation. Effective in subsequent {@link #render(SurfaceTexture)} calls.
   */
  public void setRotation(int rotation) {
    this.rotation = rotation;
  }

  /**
   * Configures whether the renderer should do an explicit CPU synchronization using glFinish upon
   * each {@link #render} call. Defaults to true.
   */
  public void setDoExplicitCpuSync(boolean doExplicitCpuSync) {
    this.doExplicitCpuSync = doExplicitCpuSync;
  }

  /**
   * Renders the surfaceTexture to the framebuffer with optional vertical flip.
   *
   * <p>Before calling this, {@link #setup} must have been called.
   *
   * <p>NOTE: Calls {@link SurfaceTexture#updateTexImage()} on passed surface texture.
   */
  public void render(SurfaceTexture surfaceTexture) {
    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

    GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
    ShaderUtil.checkGlError("glActiveTexture");
    surfaceTexture.updateTexImage(); // This implicitly binds the texture.
    surfaceTexture.getTransformMatrix(textureTransformMatrix);
    GLES20.glTexParameteri(
        GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
    GLES20.glTexParameteri(
        GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
    GLES20.glTexParameteri(
        GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
    GLES20.glTexParameteri(
        GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
    ShaderUtil.checkGlError("glTexParameteri");

    GLES20.glUseProgram(program);
    ShaderUtil.checkGlError("glUseProgram");
    GLES20.glUniform1i(frameUniform, 0);
    ShaderUtil.checkGlError("glUniform1i");
    GLES20.glUniformMatrix4fv(textureTransformUniform, 1, false, textureTransformMatrix, 0);
    ShaderUtil.checkGlError("glUniformMatrix4fv");
    GLES20.glEnableVertexAttribArray(ATTRIB_POSITION);
    GLES20.glVertexAttribPointer(
        ATTRIB_POSITION, 2, GLES20.GL_FLOAT, false, 0, getPositionVerticies());

    GLES20.glEnableVertexAttribArray(ATTRIB_TEXTURE_COORDINATE);
    GLES20.glVertexAttribPointer(
        ATTRIB_TEXTURE_COORDINATE,
        2,
        GLES20.GL_FLOAT,
        false,
        0,
        flipY ? FLIPPED_TEXTURE_VERTICES : TEXTURE_VERTICES);
    ShaderUtil.checkGlError("program setup");

    GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
    ShaderUtil.checkGlError("glDrawArrays");
    GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0);
    ShaderUtil.checkGlError("glBindTexture");

    if (doExplicitCpuSync) {

      // TODO: add sync and go back to glFlush()
      GLES20.glFinish();
    }
  }

  /**
   * Call this to delete the shader program.
   *
   * <p>This is only necessary if one wants to release the program while keeping the context around.
   */
  public void release() {
    GLES20.glDeleteProgram(program);
  }

  private FloatBuffer getPositionVerticies() {
    switch (rotation) {
      case Surface.ROTATION_90:
        return POSITION_VERTICIES_90;
      case Surface.ROTATION_180:
        return POSITION_VERTICIES_180;
      case Surface.ROTATION_270:
        return POSITION_VERTICIES_270;
      case Surface.ROTATION_0:
      default:
        return POSITION_VERTICIES_0;
    }
  }

  private static FloatBuffer fb(Vertex[] v, int i0, int i1, int i2, int i3) {
    return ShaderUtil.floatBuffer(
        v[i0].x, v[i0].y, v[i1].x, v[i1].y, v[i2].x, v[i2].y, v[i3].x, v[i3].y);
  }

  /** Convenience class to make rotations easier. */
  private static class Vertex {
    float x;
    float y;

    Vertex(float x, float y) {
      this.x = x;
      this.y = y;
    }
  }
}
