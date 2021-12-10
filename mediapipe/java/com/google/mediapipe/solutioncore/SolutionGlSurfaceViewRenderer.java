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

package com.google.mediapipe.solutioncore;

import android.graphics.SurfaceTexture;
import android.opengl.GLES20;
import android.opengl.Matrix;
import com.google.mediapipe.components.GlSurfaceViewRenderer;
import com.google.mediapipe.framework.TextureFrame;
import com.google.mediapipe.glutil.ShaderUtil;
import java.util.concurrent.atomic.AtomicReference;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * MediaPipe Solution's GlSurfaceViewRenderer.
 *
 * <p>Users can provide a custom {@link ResultGlRenderer} for rendering MediaPipe solution results.
 * For setting the latest solution result, call {@link #setRenderData(ImageSolutionResult,
 * boolean)}. By default, the renderer renders the input images. Call {@link
 * #setRenderInputImage(boolean)} to explicitly set whether the input images should be rendered or
 * not.
 */
public class SolutionGlSurfaceViewRenderer<T extends ImageSolutionResult>
    extends GlSurfaceViewRenderer {
  private static final String TAG = "SolutionGlSurfaceViewRenderer";
  private boolean renderInputImage = true;
  private final AtomicReference<T> nextSolutionResult = new AtomicReference<>();
  private ResultGlRenderer<T> resultGlRenderer;

  /** Sets if the input image needs to be rendered. Default to true. */
  public void setRenderInputImage(boolean renderInputImage) {
    this.renderInputImage = renderInputImage;
  }

  /** Sets a user-defined {@link ResultGlRenderer} for rendering MediaPipe solution results. */
  public void setSolutionResultRenderer(ResultGlRenderer<T> resultGlRenderer) {
    this.resultGlRenderer = resultGlRenderer;
  }

  /**
   * Sets the next input {@link TextureFrame} and solution result to render.
   *
   * @param solutionResult a solution result object that contains the solution outputs and a
   *     textureframe.
   * @param produceTextureFrames whether to produce and cache all the {@link TextureFrame}s for
   *     further use.
   */
  public void setRenderData(T solutionResult, boolean produceTextureFrames) {
    TextureFrame frame = solutionResult.acquireInputTextureFrame();
    setFrameSize(frame.getWidth(), frame.getHeight());
    setNextFrame(frame);
    if (produceTextureFrames) {
      solutionResult.produceAllTextureFrames();
    }
    T oldSolutionResult = nextSolutionResult.getAndSet(solutionResult);
    if (oldSolutionResult != null) {
      oldSolutionResult.releaseCachedTextureFrames();
    }
  }

  @Override
  public void onSurfaceCreated(GL10 gl, EGLConfig config) {
    super.onSurfaceCreated(gl, config);
    resultGlRenderer.setupRendering();
  }

  @Override
  public void onDrawFrame(GL10 gl) {
    TextureFrame frame = null;
    if (renderInputImage) {
      frame = renderFrame();
    } else {
      GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
      ShaderUtil.checkGlError("glClear");
      GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
      ShaderUtil.checkGlError("glActiveTexture");
    }
    T solutionResult = null;
    if (nextSolutionResult != null) {
      solutionResult = nextSolutionResult.getAndSet(null);
      float[] textureBoundary = calculateTextureBoundary();
      float[] projectionMatrix = new float[16];
      // See {@link ResultGlRenderer#renderResult}.
      Matrix.orthoM(
          projectionMatrix, /* result */
          0, /* offset */
          textureBoundary[0], /* left */
          textureBoundary[1], /* right */
          textureBoundary[3], /* bottom */
          textureBoundary[2], /* top */
          -1, /* near */
          1 /* far */);
      resultGlRenderer.renderResult(solutionResult, projectionMatrix);
    }
    flush(frame);
    if (solutionResult != null) {
      solutionResult.releaseCachedTextureFrames();
    }
  }

  @Override
  public void setSurfaceTexture(SurfaceTexture texture) {
    throw new IllegalStateException("SurfaceTexture should not be used in MediaPipe Solution.");
  }
}
