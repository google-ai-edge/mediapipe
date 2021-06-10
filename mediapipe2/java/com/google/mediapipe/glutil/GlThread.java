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

import android.opengl.GLES20;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import javax.annotation.Nullable;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLSurface;

/**
 * A thread that manages an OpenGL context.
 *
 * <p>A given context can only be used by a single thread at a time. Furthermore, on at least some
 * Android devices, changing the current context is reported to take a long time (on the order of
 * several ms). Therefore it is often convenient to have a dedicated thread for rendering to a
 * context.
 */
public class GlThread extends Thread {
  private static final String TAG = "GlThread";
  private static final String THREAD_NAME = "mediapipe.glutil.GlThread";

  private boolean doneStarting;
  private boolean startedSuccessfully;
  private final Object startLock = new Object();

  protected volatile EglManager eglManager;
  protected EGLSurface eglSurface = null;
  protected Handler handler = null;  // must be created on the thread itself
  protected Looper looper = null; // must be created on the thread itself
  protected int framebuffer = 0;

  /**
   * Creates a GlThread.
   *
   * @param parentContext another EGL context with which to share data (e.g. textures); can be an
   *     {@link EGLContext} or an {@link android.opengl.EGLContext}; can be null.
   */
  public GlThread(@Nullable Object parentContext) {
    this(parentContext, null);
  }

  /**
   * Creates a GlThread.
   *
   * @param parentContext another EGL context with which to share data (e.g. textures); can be an
   *     {@link EGLContext} or an {@link android.opengl.EGLContext}; can be null.
   * @param additionalConfigAttributes a list of attributes for eglChooseConfig to be added to the
   *     default ones.
   */
  public GlThread(@Nullable Object parentContext, @Nullable int[] additionalConfigAttributes) {
    eglManager = new EglManager(parentContext, additionalConfigAttributes);
    setName(THREAD_NAME);
  }

  /**
   * Returns the Handler associated with this thread.
   */
  public Handler getHandler() {
    return handler;
  }

  /** Returns the Looper associated with this thread. */
  public Looper getLooper() {
    return looper;
  }

  /** Returns the EglManager managing this thread's context. */
  public EglManager getEglManager() {
    return eglManager;
  }

  /**
   * Returns the EGLContext associated with this thread. Do not use it on another thread.
   */
  public EGLContext getEGLContext() {
    return eglManager.getContext();
  }

  /**
   * Returns the framebuffer object used by this thread.
   */
  public int getFramebuffer() {
    return framebuffer;
  }

  /**
   * Binds a texture to the color attachment of the framebuffer.
   */
  public void bindFramebuffer(int texture, int width, int height) {
    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, framebuffer);
    GLES20.glFramebufferTexture2D(
        GLES20.GL_FRAMEBUFFER,
        GLES20.GL_COLOR_ATTACHMENT0,
        GLES20.GL_TEXTURE_2D,
        texture,
        0);
    int status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
    if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
      throw new RuntimeException("Framebuffer not complete, status=" + status);
    }
    GLES20.glViewport(0, 0, width, height);
    ShaderUtil.checkGlError("glViewport");
  }

  @Override
  public void run() {
    try {
      Looper.prepare();
      handler = createHandler();
      looper = Looper.myLooper();

      Log.d(TAG, String.format("Starting GL thread %s", getName()));

      prepareGl();
      startedSuccessfully = true;
    } finally {
      // Always stop waitUntilReady here, even if we got an exception.
      // Otherwise the main thread may be stuck waiting.
      synchronized (startLock) {
        doneStarting = true;
        startLock.notify(); // signal waitUntilReady()
      }
    }

    try {
      Looper.loop();
    } finally {
      looper = null;

      releaseGl();
      eglManager.release();

      Log.d(TAG, String.format("Stopping GL thread %s", getName()));
    }
  }

  /** Terminates the thread, after processing all pending messages. */
  public boolean quitSafely() {
    if (looper == null) {
      return false;
    }
    looper.quitSafely();
    return true;
  }

  /**
   * Waits until the thread has finished setting up the handler and invoking {@link prepareGl}.
   * Returns true if no exceptions were raised during the process.
   */
  public boolean waitUntilReady() throws InterruptedException {
    // We wait in a loop to deal with spurious wakeups. However, we do not
    // catch the InterruptedException, because we have no way of knowing what
    // the application expects. On one hand, the called expects the thread to
    // be ready when this method returns, which means we would have to keep
    // looping. But on the other hand, if they interrupt the thread they may
    // not want it to continue execution. We have no choice but to propagate
    // the exception and let the caller make the decision.
    synchronized (startLock) {
      while (!doneStarting) {
        startLock.wait();
      }
    }
    return startedSuccessfully;
  }

  /** Sets up the OpenGL context. Can be overridden to set up additional resources. */
  public void prepareGl() {
    eglSurface = createEglSurface();
    eglManager.makeCurrent(eglSurface, eglSurface);

    GLES20.glDisable(GLES20.GL_DEPTH_TEST);
    GLES20.glDisable(GLES20.GL_CULL_FACE);
    int[] values = new int[1];
    GLES20.glGenFramebuffers(1, values, 0);
    framebuffer = values[0];
  }

  /** Releases the resources created in prepareGl. */
  public void releaseGl() {
    if (framebuffer != 0) {
      int[] values = new int[1];
      values[0] = framebuffer;
      GLES20.glDeleteFramebuffers(1, values, 0);
      framebuffer = 0;
    }

    eglManager.makeNothingCurrent();
    if (eglSurface != null) {
      eglManager.releaseSurface(eglSurface);
      eglSurface = null;
    }
  }

  /**
   * Factory method that creates the handler used by the thread. Can be overridden to use a custom
   * {@link Handler}.
   */
  protected Handler createHandler() {
    return new Handler();
  }

  /** Factory method that creates the surface used by the thread. */
  protected EGLSurface createEglSurface() {
    return eglManager.createOffscreenSurface(1, 1);
  }
}
