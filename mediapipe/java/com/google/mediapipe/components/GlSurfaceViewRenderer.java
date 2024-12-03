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
import static java.lang.Math.min;

import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLES31;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.util.Log;
import android.util.Pair;
import com.google.mediapipe.framework.TextureFrame;
import com.google.mediapipe.glutil.CommonShaders;
import com.google.mediapipe.glutil.ShaderUtil;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
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
  /** Listener for image capture requests. */
  public interface ImageCaptureListener {
    void onImageCaptured(int width, int height, int[] data);
  }

  /**
   * Scale to use when the frame and view have different aspect ratios.
   */
  public enum Scale {
    FILL,
    FIT,
    FIT_TO_WIDTH,
    FIT_TO_HEIGHT
  };

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
  private final AtomicBoolean captureNextFrameImage = new AtomicBoolean();
  private ImageCaptureListener imageCaptureListener;
  // Specifies whether a black CLAMP_TO_BORDER effect should be used.
  private boolean shouldClampToBorder = false;
  private Scale scale = Scale.FILL;

  private float zoomFactor = 1.0f;
  private Pair<Float, Float> zoomLocation = new Pair<>(0.5f, 0.5f);

  /** Sets the {@link ImageCaptureListener}. */
  public void setImageCaptureListener(ImageCaptureListener imageCaptureListener) {
    this.imageCaptureListener = imageCaptureListener;
  }

  /**
   * Request to capture Image of the next frame.
   *
   * <p>The result will be provided to the {@link ImageCaptureListener} if one is set. Please note
   * this is an expensive operation and the result may not be available for a while.
   */
  public void captureNextFrameImage() {
    captureNextFrameImage.set(true);
  }

  @Override
  public void onSurfaceCreated(GL10 gl, EGLConfig config) {
    if (surfaceTexture == null) {
      Matrix.setIdentityM(textureTransformMatrix, 0 /* offset */);
    }
    Map<String, Integer> attributeLocations = new HashMap<>();
    attributeLocations.put("position", ATTRIB_POSITION);
    attributeLocations.put("texture_coordinate", ATTRIB_TEXTURE_COORDINATE);
    Log.d(TAG, "external texture: " + isExternalTexture());
    String fragmentShader;
    if (shouldClampToBorder) {
      fragmentShader = isExternalTexture()
          ? CommonShaders.FRAGMENT_SHADER_EXTERNAL_CLAMP_TO_BORDER
          : CommonShaders.FRAGMENT_SHADER_CLAMP_TO_BORDER;
    } else {
      fragmentShader = isExternalTexture()
          ? CommonShaders.FRAGMENT_SHADER_EXTERNAL
          : CommonShaders.FRAGMENT_SHADER;
    }
    program =
        ShaderUtil.createProgram(
            CommonShaders.VERTEX_SHADER,
            fragmentShader,
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

    // Capture image if requested.
    ImageCaptureListener imageCaptureListener = this.imageCaptureListener;
    if (captureNextFrameImage.getAndSet(false) && imageCaptureListener != null) {
      // Find the name of the bound texture.
      int[] texName = new int[1];
      if (surfaceTexture != null) {
        GLES20.glGetIntegerv(GLES11Ext.GL_TEXTURE_BINDING_EXTERNAL_OES, texName, 0);
      } else {
        texName[0] = frame.getTextureName();
      }

      // Find texture size and create byte buffer large enough to hold an RGBA image.
      int[] texDims = new int[2];
      GLES31.glGetTexLevelParameteriv(textureTarget, 0, GLES31.GL_TEXTURE_WIDTH, texDims, 0);
      GLES31.glGetTexLevelParameteriv(textureTarget, 0, GLES31.GL_TEXTURE_HEIGHT, texDims, 1);
      int texWidth = texDims[0];
      int texHeight = texDims[1];
      int imageSize = texWidth * texHeight;
      ByteBuffer byteBuffer = ByteBuffer.allocateDirect(imageSize * 4);
      byteBuffer.order(ByteOrder.nativeOrder());

      // Read pixels from texture.
      int[] fbo = new int[1];
      GLES20.glGenFramebuffers(1, fbo, 0);
      GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo[0]);
      GLES20.glFramebufferTexture2D(
          GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, texName[0], 0);
      GLES20.glReadPixels(
          0, 0, texWidth, texHeight, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, byteBuffer);
      GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
      GLES20.glDeleteFramebuffers(1, fbo, 0);
      ShaderUtil.checkGlError("capture frame");

      int[] data = new int[imageSize];
      byteBuffer.asIntBuffer().get(data);
      imageCaptureListener.onImageCaptured(texWidth, texHeight, data);
    }

    GLES20.glBindTexture(textureTarget, 0);
    ShaderUtil.checkGlError("unbind surfaceTexture");

    return frame;
  }

  /** Returns the texture left, right, bottom, and top visible boundaries. */
  public float[] calculateTextureBoundary() {
    // TODO: compute scale from surfaceTexture size.
    float scaleWidth = frameWidth > 0 ? (float) surfaceWidth / (float) frameWidth : 1.0f;
    float scaleHeight = frameHeight > 0 ? (float) surfaceHeight / (float) frameHeight : 1.0f;

    // By default, FILL setting is used. That is, whichever of the two scales is greater corresponds
    // to the dimension where the image is proportionally smaller than the view. Dividing both
    // scales by that number results in that dimension having scale 1.0, and thus touching the edges
    // of the view, while the other is cropped proportionally.
    float scale = max(scaleWidth, scaleHeight);

    // If any of the FIT settings are used, the min scale is divided so the frame doesn't exceed
    // the view in that direction (or both for FIT).
    if (this.scale == Scale.FIT
        || (this.scale == Scale.FIT_TO_WIDTH && (frameWidth > frameHeight))
        || (this.scale == Scale.FIT_TO_HEIGHT && (frameHeight > frameWidth))) {
      scale = min(scaleWidth, scaleHeight);
    }

    scaleWidth /= scale;
    scaleHeight /= scale;

    // Alignment controls where the visible section is placed within the full camera frame, with
    // (0, 0) being the bottom left, and (1, 1) being the top right.
    float textureLeft = (1.0f - scaleWidth) * alignmentHorizontal;
    float textureRight = textureLeft + scaleWidth;
    float textureBottom = (1.0f - scaleHeight) * alignmentVertical;
    float textureTop = textureBottom + scaleHeight;

    Pair<Float, Float> currentZoomLocation = this.zoomLocation;
    float zoomLocationX = currentZoomLocation.first;
    float zoomLocationY = currentZoomLocation.second;

    textureLeft = (textureLeft - 0.5f) / zoomFactor + zoomLocationX;
    textureRight = (textureRight - 0.5f) / zoomFactor + zoomLocationX;
    textureBottom = (textureBottom - 0.5f) / zoomFactor + zoomLocationY;
    textureTop = (textureTop - 0.5f) / zoomFactor + zoomLocationY;

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
  public void setNextFrame(@Nullable TextureFrame frame) {
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

  /**
   * Whether to use GL_CLAMP_TO_BORDER-like mode. This is useful when rendering landscape or
   * different aspect ratio frames. The remaining area will be rendered black.
   */
  public void setClampToBorder(boolean shouldClampToBorder) {
    if (program != 0) {
      throw new IllegalStateException(
          "setClampToBorder must be called before the surface is created");
    }
    this.shouldClampToBorder = shouldClampToBorder;
  }

  /** {@link Scale} option to use when frame and view have different aspect ratios. */
  public void setScale(Scale scale) {
    this.scale = scale;
  }

  /** Zoom factor applied to the frame, must not be 0. */
  public void setZoomFactor(float zoomFactor) {
    if (zoomFactor == 0.f) {
      return;
    }
    this.zoomFactor = zoomFactor;
  }

  /**
   * Location where to apply the zooming of the frame to. Default is 0.5, 0.5 (scaling is applied to
   * the center).
   */
  public void setZoomLocation(float zoomLocationX, float zoomLocationY) {
    this.zoomLocation = new Pair<>(zoomLocationX, zoomLocationY);
  }

  private boolean isExternalTexture() {
    return textureTarget == GLES11Ext.GL_TEXTURE_EXTERNAL_OES;
  }
}
