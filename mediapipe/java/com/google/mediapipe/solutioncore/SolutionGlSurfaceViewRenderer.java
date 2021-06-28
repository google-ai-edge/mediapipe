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
import com.google.mediapipe.components.GlSurfaceViewRenderer;
import com.google.mediapipe.framework.TextureFrame;
import java.util.concurrent.atomic.AtomicReference;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * MediaPipe Solution's GlSurfaceViewRenderer.
 *
 * <p>Users can provide a custom {@link ResultGlRenderer} for rendering MediaPipe solution results.
 * For setting the latest solution result, call {@link #setRenderData(ImageSolutionResult)}. By
 * default, the renderer renders the input images. Call {@link #setRenderInputImage(boolean)} to
 * explicitly set whether the input images should be rendered or not.
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
   * Sets the next textureframe and solution result to render.
   *
   * @param solutionResult a solution result object that contains the solution outputs and a
   *     textureframe.
   */
  public void setRenderData(T solutionResult) {
    setNextFrame(solutionResult.acquireTextureFrame());
    nextSolutionResult.getAndSet(solutionResult);
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
    }
    if (nextSolutionResult != null) {
      T solutionResult = nextSolutionResult.getAndSet(null);
      resultGlRenderer.renderResult(solutionResult);
    }
    flush(frame);
  }

  @Override
  public void setSurfaceTexture(SurfaceTexture texture) {
    throw new IllegalStateException("SurfaceTexture should not be used in MediaPipe Solution.");
  }
}
