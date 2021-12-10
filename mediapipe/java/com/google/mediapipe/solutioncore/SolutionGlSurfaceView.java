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

import android.content.Context;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.View;
import com.google.mediapipe.glutil.EglManager;
import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;

/**
 * A simplified GlSurfaceView implementation for displaying MediaPipe Solution results.
 *
 * <p>Users need to provide a custom {@link ResultGlRenderer} via {@link
 * setSolutionResultRenderer(ResultGlRenderer)} for rendering MediaPipe solution results. Setting
 * the latest render data by calling {@link #setRenderData(ImageSolutionResult)} before invoking
 * {@link #requestRender}. By default, the solution renderer renders the input images. Call {@link
 * #setRenderInputImage(boolean)} to explicitly set whether the input images should be rendered or
 * not.
 */
public class SolutionGlSurfaceView<T extends ImageSolutionResult> extends GLSurfaceView {
  private static final String TAG = "SolutionGlSurfaceView";
  SolutionGlSurfaceViewRenderer<T> renderer = new SolutionGlSurfaceViewRenderer<>();

  /**
   * Sets a user-defined {@link ResultGlRenderer} for rendering MediaPipe solution results.
   *
   * @param resultRenderer a {@link ResultGlRenderer}.
   */
  public void setSolutionResultRenderer(ResultGlRenderer<T> resultRenderer) {
    renderer.setSolutionResultRenderer(resultRenderer);
  }

  /**
   * Sets the next input {@link TextureFrame} and solution result to render.
   *
   * @param solutionResult a solution result object that contains the solution outputs and a
   *     textureframe.
   */
  public void setRenderData(T solutionResult) {
    renderer.setRenderData(solutionResult, false);
  }

  /**
   * Sets the next input {@link TextureFrame} and solution result to render.
   *
   * @param solutionResult a solution result object that contains the solution outputs and a {@link
   *     TextureFrame}.
   * @param produceTextureFrames whether to produce and cache all the {@link TextureFrame}s for
   *     further use.
   */
  public void setRenderData(T solutionResult, boolean produceTextureFrames) {
    renderer.setRenderData(solutionResult, produceTextureFrames);
  }

  /** Sets if the input image needs to be rendered. Default to true. */
  public void setRenderInputImage(boolean renderInputImage) {
    renderer.setRenderInputImage(renderInputImage);
  }

  /** Initializes SolutionGlSurfaceView with Android context, gl context, and gl version number. */
  public SolutionGlSurfaceView(Context context, EGLContext glContext, int glMajorVersion) {
    super(context);
    setEGLContextClientVersion(glMajorVersion);
    getHolder().addCallback(new DummyHolderCallback());
    setEGLContextFactory(
        new GLSurfaceView.EGLContextFactory() {
          @Override
          public EGLContext createContext(EGL10 egl, EGLDisplay display, EGLConfig eglConfig) {
            int[] contextAttrs = {
              EglManager.EGL_CONTEXT_CLIENT_VERSION, glMajorVersion, EGL10.EGL_NONE
            };
            return egl.eglCreateContext(display, eglConfig, glContext, contextAttrs);
          }

          @Override
          public void destroyContext(EGL10 egl, EGLDisplay display, EGLContext context) {
            if (!egl.eglDestroyContext(display, context)) {
              throw new RuntimeException("eglDestroyContext failed");
            }
          }
        });
    renderer.setTextureTarget(GLES20.GL_TEXTURE_2D);
    super.setRenderer(renderer);
    setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
    setVisibility(View.GONE);
  }

  private class DummyHolderCallback implements SurfaceHolder.Callback {
    @Override
    public void surfaceCreated(SurfaceHolder holder) {
      Log.d(TAG, "main surfaceCreated");
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
      Log.d(
          TAG,
          String.format(
              "main surfaceChanged. width: %d height: %d glViewWidth: %d glViewHeight: %d",
              width,
              height,
              SolutionGlSurfaceView.this.getWidth(),
              SolutionGlSurfaceView.this.getHeight()));
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
      Log.d(TAG, "main surfaceDestroyed");
    }
  }
}
