// Copyright 2019-2021 The MediaPipe Authors.
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

package com.google.mediapipe.components;

import static java.lang.Math.max;

import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.util.Log;
import com.google.mediapipe.framework.TextureFrame;
import com.google.mediapipe.glutil.CommonShaders;
import com.google.mediapipe.glutil.ShaderUtil;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Renderer for a {@link GLSurfaceView}. It displays a texture. The texture is scaled and cropped as
 * necessary to fill the view, while maintaining its aspect ratio.
 *
 * <p>It can render both textures bindable to the normal {@link GLES20#GL_TEXTURE_2D} target as well
 * as textures bindable to {@link GLES11Ext#GL_TEXTURE_EXTERNAL_OES}, which is used for Android
 * surfaces. Call {@link #setTextureTarget(int)} to choose the correct target.
 *
 * <p>It can display a {@link SurfaceTexture} (call {@link #setSurfaceTexture(SurfaceTexture)}) or a
 * {@link TextureFrame} (call {@link #setNextFrame(TextureFrame)}).
 */
public class GlSurfaceViewRenderer implements GLSurfaceView.Renderer {
  private static final String TAG = "DemoRenderer";
  private static final int ATTRIB_POSITION = 1;
  private static final int ATTRIB_TEXTURE_COORDINATE = 2;

  private int surfaceWidth;
  private int surfaceHeight;
  private int frameWidth = 0;
  private int frameHeight = 0;
  private int program = 0;
  private int frameUniform;
  private int textureTarget = GLES11Ext.GL_TEXTURE_EXTERNAL_OES;
  private int textureTransformUniform;
  // Controls the alignment between frame size and surface size, 0.5f default is centered.
  private float alignmentHorizontal = 0.5f;
  private float alignmentVertical = 0.5f;
  private float[] textureTransformMatrix = new float[16];
  private SurfaceTexture surfaceTexture = null;
  private final AtomicReference<TextureFrame> nextFrame = new AtomicReference<>();

  @Override
  public void onSurfaceCreated(GL10 gl, EGLConfig config) {
    if (surfaceTexture == null) {
      Matrix.setIdentityM(textureTransformMatrix, 0 /* offset */);
    }
    Map<String, Integer> attributeLocations = new HashMap<>();
    attributeLocations.put("position", ATTRIB_POSITION);
    attributeLocations.put("texture_coordinate", ATTRIB_TEXTURE_COORDINATE);
    Log.d(TAG, "external texture: " + isExternalTexture());
    program =
        ShaderUtil.createProgram(
            CommonShaders.VERTEX_SHADER,
            isExternalTexture()
                ? CommonShaders.FRAGMENT_SHADER_EXTERNAL
                : CommonShaders.FRAGMENT_SHADER,
            attributeLocations);
    frameUniform = GLES20.glGetUniformLocation(program, "video_frame");
    textureTransformUniform = GLES20.glGetUniformLocation(program, "texture_transform");
    ShaderUtil.checkGlError("glGetUniformLocation");

    GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  }

  @Override
  public void onSurfaceChanged(GL10 gl, int width, int height) {
    surfaceWidth = width;
    surfaceHeight = height;
    GLES20.glViewport(0, 0, width, height);
  }

  /** Renders the frame. Note that the {@link #flush} method must be called afterwards. */
  protected TextureFrame renderFrame() {
    TextureFrame frame = nextFrame.getAndSet(null);

    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
    ShaderUtil.checkGlError("glClear");

    if (surfaceTexture == null && frame == null) {
      return null;
    }

    GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
    ShaderUtil.checkGlError("glActiveTexture");
    if (surfaceTexture != null) {
      surfaceTexture.updateTexImage();
      surfaceTexture.getTransformMatrix(textureTransformMatrix);
    } else {
      GLES20.glBindTexture(textureTarget, frame.getTextureName());
      ShaderUtil.checkGlError("glBindTexture");
    }
    GLES20.glTexParameteri(textureTarget, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
    GLES20.glTexParameteri(textureTarget, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
    GLES20.glTexParameteri(textureTarget, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
    GLES20.glTexParameteri(textureTarget, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
    ShaderUtil.checkGlError("texture setup");

    GLES20.glUseProgram(program);
    GLES20.glUniform1i(frameUniform, 0);
    GLES20.glUniformMatrix4fv(textureTransformUniform, 1, false, textureTransformMatrix, 0);
    ShaderUtil.checkGlError("glUniformMatrix4fv");
    GLES20.glEnableVertexAttribArray(ATTRIB_POSITION);
    GLES20.glVertexAttribPointer(
        ATTRIB_POSITION, 2, GLES20.GL_FLOAT, false, 0, CommonShaders.SQUARE_VERTICES);

    float[] boundary = calculateTextureBoundary();
    float textureLeft = boundary[0];
    float textureRight = boundary[1];
    float textureBottom = boundary[2];
    float textureTop = boundary[3];
    // Unlike on iOS, there is no need to flip the surfaceTexture here.
    // But for regular textures, we will need to flip them.
    final FloatBuffer passThroughTextureVertices =
        ShaderUtil.floatBuffer(
            textureLeft, textureBottom,
            textureRight, textureBottom,
            textureLeft, textureTop,
            textureRight, textureTop);
    GLES20.glEnableVertexAttribArray(ATTRIB_TEXTURE_COORDINATE);
    GLES20.glVertexAttribPointer(
        ATTRIB_TEXTURE_COORDINATE, 2, GLES20.GL_FLOAT, false, 0, passThroughTextureVertices);
    ShaderUtil.checkGlError("program setup");

    GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
    ShaderUtil.checkGlError("glDrawArrays");
    GLES20.glBindTexture(textureTarget, 0);
    ShaderUtil.checkGlError("unbind surfaceTexture");

    return frame;
  }

  /** Returns the texture left, right, bottom, and top visible boundaries. */
  protected float[] calculateTextureBoundary() {
    // TODO: compute scale from surfaceTexture size.
    float scaleWidth = frameWidth > 0 ? (float) surfaceWidth / (float) frameWidth : 1.0f;
    float scaleHeight = frameHeight > 0 ? (float) surfaceHeight / (float) frameHeight : 1.0f;
    // Whichever of the two scales is greater corresponds to the dimension where the image
    // is proportionally smaller than the view. Dividing both scales by that number results
    // in that dimension having scale 1.0, and thus touching the edges of the view, while the
    // other is cropped proportionally.
    float maxScale = max(scaleWidth, scaleHeight);
    scaleWidth /= maxScale;
    scaleHeight /= maxScale;

    // Alignment controls where the visible section is placed within the full camera frame, with
    // (0, 0) being the bottom left, and (1, 1) being the top right.
    float textureLeft = (1.0f - scaleWidth) * alignmentHorizontal;
    float textureRight = textureLeft + scaleWidth;
    float textureBottom = (1.0f - scaleHeight) * alignmentVertical;
    float textureTop = textureBottom + scaleHeight;

    return new float[] {textureLeft, textureRight, textureBottom, textureTop};
  }

  /**
   * Calls {@link GLES20.glFlush} and releases the texture frame. Should be invoked after the {@link
   * #renderFrame} method is called.
   *
   * @param frame the {@link TextureFrame} to be released after {@link GLES20.glFlush}.
   */
  protected void flush(TextureFrame frame) {
    GLES20.glFlush();
    if (frame != null) {
      frame.release();
    }
  }

  @Override
  public void onDrawFrame(GL10 gl) {
    TextureFrame frame = renderFrame();
    flush(frame);
  }

  public void setTextureTarget(int target) {
    if (program != 0) {
      throw new IllegalStateException(
          "setTextureTarget must be called before the surface is created");
    }
    textureTarget = target;
  }

  public void setSurfaceTexture(SurfaceTexture texture) {
    if (!isExternalTexture()) {
      throw new IllegalStateException(
          "to use a SurfaceTexture, the texture target must be GL_TEXTURE_EXTERNAL_OES");
    }
    TextureFrame oldFrame = nextFrame.getAndSet(null);
    if (oldFrame != null) {
      oldFrame.release();
    }
    surfaceTexture = texture;
  }

  // Use this when the texture is not a SurfaceTexture.
  public void setNextFrame(TextureFrame frame) {
    if (surfaceTexture != null) {
      Matrix.setIdentityM(textureTransformMatrix, 0 /* offset */);
    }
    TextureFrame oldFrame = nextFrame.getAndSet(frame);
    if (oldFrame != null && oldFrame != frame) {
      oldFrame.release();
    }
    surfaceTexture = null;
  }

  public void setFrameSize(int width, int height) {
    frameWidth = width;
    frameHeight = height;
  }

  /**
   * When the aspect ratios between the camera frame and the surface size are mismatched, this
   * controls how the image is aligned. 0.0 means aligning the left/bottom edges; 1.0 means aligning
   * the right/top edges; 0.5 (default) means aligning the centers.
   */
  public void setAlignment(float horizontal, float vertical) {
    alignmentHorizontal = horizontal;
    alignmentVertical = vertical;
  }

  private boolean isExternalTexture() {
    return textureTarget == GLES11Ext.GL_TEXTURE_EXTERNAL_OES;
  }
}
