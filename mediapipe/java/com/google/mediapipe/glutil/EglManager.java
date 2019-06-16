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
import android.opengl.EGL14;
import android.os.Build;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import com.google.mediapipe.framework.Compat;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;
import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.egl.EGLSurface;

/**
 * Helper class for creating and managing an {@link EGLContext}.
 *
 * <p>Note: Since we want to support API level 16, we cannot rely on {@link android.opengl.EGL14}.
 */
public class EglManager {
  private static final String TAG = "EglManager";

  // These are missing from EGL10.
  public static final int EGL_CONTEXT_CLIENT_VERSION = 0x3098;
  public static final int EGL_OPENGL_ES2_BIT = 0x4;
  public static final int EGL_OPENGL_ES3_BIT_KHR = 0x00000040;
  public static final int EGL_DRAW = 12377;
  public static final int EGL_READ = 12378;

  public static final int EGL14_API_LEVEL = android.os.Build.VERSION_CODES.JELLY_BEAN_MR1;

  private EGL10 egl;
  private EGLDisplay eglDisplay = EGL10.EGL_NO_DISPLAY;
  private EGLConfig eglConfig = null;
  private EGLContext eglContext = EGL10.EGL_NO_CONTEXT;
  private int[] singleIntArray;  // reuse this instead of recreating it each time
  private int glVersion;
  private long nativeEglContext = 0;
  private android.opengl.EGLContext egl14Context = null;

  /**
   * Creates an EglManager wrapping a new {@link EGLContext}.
   *
   * @param parentContext another EGL context with which to share data (e.g. textures); can be an
   *     {@link EGLContext} or an {@link android.opengl.EGLContext}; can be null.
   */
  public EglManager(@Nullable Object parentContext) {
    this(parentContext, null);
  }

  /**
   * Creates an EglManager wrapping a new {@link EGLContext}.
   *
   * @param parentContext another EGL context with which to share data (e.g. textures); can be an
   *     {@link EGLContext} or an {@link android.opengl.EGLContext}; can be null.
   * @param additionalConfigAttributes a list of attributes for eglChooseConfig to be added to the
   *     default ones.
   */
  public EglManager(@Nullable Object parentContext, @Nullable int[] additionalConfigAttributes) {
    singleIntArray = new int[1];
    egl = (EGL10) EGLContext.getEGL();
    eglDisplay = egl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY);
    if (eglDisplay == EGL10.EGL_NO_DISPLAY) {
      throw new RuntimeException("eglGetDisplay failed");
    }
    int[] version = new int[2];
    if (!egl.eglInitialize(eglDisplay, version)) {
      throw new RuntimeException("eglInitialize failed");
    }

    EGLContext realParentContext;
    if (parentContext == null) {
      realParentContext = EGL10.EGL_NO_CONTEXT;
    } else if (parentContext instanceof EGLContext) {
      realParentContext = (EGLContext) parentContext;
    } else if (Build.VERSION.SDK_INT >= EGL14_API_LEVEL
        && parentContext instanceof android.opengl.EGLContext) {
      if (parentContext == EGL14.EGL_NO_CONTEXT) {
        realParentContext = EGL10.EGL_NO_CONTEXT;
      } else {
        realParentContext = egl10ContextFromEgl14Context((android.opengl.EGLContext) parentContext);
      }
    } else {
      throw new RuntimeException("invalid parent context: " + parentContext);
    }

    // Try to create an OpenGL ES 3 context first, then fall back on ES 2.
    try {
      createContext(realParentContext, 3, additionalConfigAttributes);
      glVersion = 3;
    } catch (RuntimeException e) {
      Log.w(TAG, "could not create GLES 3 context: " + e);
      createContext(realParentContext, 2, additionalConfigAttributes);
      glVersion = 2;
    }
  }

  /** Returns the managed {@link EGLContext} */
  public EGLContext getContext() {
    return eglContext;
  }

  /** Returns the native handle to the context. */
  public long getNativeContext() {
    if (nativeEglContext == 0) {
      grabContextVariants();
    }
    return nativeEglContext;
  }

  public android.opengl.EGLContext getEgl14Context() {
    if (Build.VERSION.SDK_INT < EGL14_API_LEVEL) {
      throw new RuntimeException("cannot use EGL14 on API level < 17");
    }
    if (egl14Context == null) {
      grabContextVariants();
    }
    return egl14Context;
  }

  public int getGlMajorVersion() {
    return glVersion;
  }

  /** Makes this the current EGL context on the current thread. */
  public void makeCurrent(EGLSurface drawSurface, EGLSurface readSurface) {
    if (!egl.eglMakeCurrent(eglDisplay, drawSurface, readSurface, eglContext)) {
      throw new RuntimeException("eglMakeCurrent failed");
    }
  }

  /** Makes no EGL context current on the current thread. */
  public void makeNothingCurrent() {
    if (!egl.eglMakeCurrent(
        eglDisplay, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_CONTEXT)) {
      throw new RuntimeException("eglMakeCurrent failed");
    }
  }

  /**
   * Creates an {@link EGLSurface} for an Android Surface.
   *
   * @param surface can be a {@link Surface}, {@link SurfaceTexture}, {@link SurfaceHolder} or
   *        {@link SurfaceView}.
   */
  public EGLSurface createWindowSurface(Object surface) {
    if (!(surface instanceof Surface
        || surface instanceof SurfaceTexture
        || surface instanceof SurfaceHolder
        || surface instanceof SurfaceView)) {
      throw new RuntimeException("invalid surface: " + surface);
    }

    // Create a window surface, and attach it to the Surface we received.
    int[] surfaceAttribs = {EGL10.EGL_NONE};
    EGLSurface eglSurface =
        egl.eglCreateWindowSurface(eglDisplay, eglConfig, surface, surfaceAttribs);
    checkEglError("eglCreateWindowSurface");
    if (eglSurface == null) {
      throw new RuntimeException("surface was null");
    }
    return eglSurface;
  }

  /**
   * Creates an {@link EGLSurface} for offscreen rendering, not bound to any Android surface.
   *
   * <p>An EGLSurface is always required to make an EGLContext current, and it is bound to the
   * OpenGL framebuffer by default. However, the framebuffer can then be bound to other objects in
   * OpenGL, such as a texture.
   * <p>If you want to use an EGLContext but do not really care about the EGLSurface, you can use
   * a 1x1 surface created with this method.
   */
  public EGLSurface createOffscreenSurface(int width, int height) {
    int[] surfaceAttribs = {EGL10.EGL_WIDTH, width, EGL10.EGL_HEIGHT, height, EGL10.EGL_NONE};
    EGLSurface eglSurface = egl.eglCreatePbufferSurface(eglDisplay, eglConfig, surfaceAttribs);
    checkEglError("eglCreatePbufferSurface");
    if (eglSurface == null) {
      throw new RuntimeException("surface was null");
    }
    return eglSurface;
  }

  /** Releases the resources held by this manager. */
  public void release() {
    if (eglDisplay != EGL10.EGL_NO_DISPLAY) {
      // Android is unusual in that it uses a reference-counted EGLDisplay.  So for
      // every eglInitialize() we need an eglTerminate().
      egl.eglMakeCurrent(
          eglDisplay, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_CONTEXT);
      egl.eglDestroyContext(eglDisplay, eglContext);
      egl.eglTerminate(eglDisplay);
    }

    eglDisplay = EGL10.EGL_NO_DISPLAY;
    eglContext = EGL10.EGL_NO_CONTEXT;
    eglConfig = null;
  }

  /** Releases an {@link EGLSurface}. */
  public void releaseSurface(EGLSurface eglSurface) {
    egl.eglDestroySurface(eglDisplay, eglSurface);
  }

  private void createContext(
      EGLContext parentContext, int glVersion, @Nullable int[] additionalConfigAttributes) {
    eglConfig = getConfig(glVersion, additionalConfigAttributes);
    if (eglConfig == null) {
      throw new RuntimeException("Unable to find a suitable EGLConfig");
    }
    // Try to create an OpenGL ES 3 context first.
    int[] contextAttrs = {EGL_CONTEXT_CLIENT_VERSION, glVersion, EGL10.EGL_NONE};
    eglContext = egl.eglCreateContext(eglDisplay, eglConfig, parentContext, contextAttrs);
    if (eglContext == null) {
      int error = egl.eglGetError();
      throw new RuntimeException(
          "Could not create GL context: EGL error: 0x"
              + Integer.toHexString(error)
              + (error == EGL10.EGL_BAD_CONTEXT
                  ? ": parent context uses a different version of OpenGL"
                  : ""));
    }
  }

  /**
   * Gets the native context handle for our context. The EGL10 API does not provide a way to do this
   * directly, but we can make our context current and grab the current context handle from native
   * code. Also gets the context as an {@link android.opengl.EGLContext}. The underlying native
   * object is always the same, but the Android API has two different wrappers for it which are
   * completely equivalent internally but completely separate at the Java level.
   */
  private void grabContextVariants() {
    EGLContext previousContext = egl.eglGetCurrentContext();
    EGLDisplay previousDisplay = egl.eglGetCurrentDisplay();
    EGLSurface previousDrawSurface = egl.eglGetCurrentSurface(EGL_DRAW);
    EGLSurface previousReadSurface = egl.eglGetCurrentSurface(EGL_READ);
    EGLSurface tempEglSurface = null;

    if (previousContext != eglContext) {
      tempEglSurface = createOffscreenSurface(1, 1);
      makeCurrent(tempEglSurface, tempEglSurface);
    }

    nativeEglContext = Compat.getCurrentNativeEGLContext();
    if (Build.VERSION.SDK_INT >= EGL14_API_LEVEL) {
      egl14Context = android.opengl.EGL14.eglGetCurrentContext();
    }

    if (previousContext != eglContext) {
      egl.eglMakeCurrent(
          previousDisplay, previousDrawSurface, previousReadSurface, previousContext);
      releaseSurface(tempEglSurface);
    }
  }

  private EGLContext egl10ContextFromEgl14Context(android.opengl.EGLContext context) {
    android.opengl.EGLContext previousContext = EGL14.eglGetCurrentContext();
    android.opengl.EGLDisplay previousDisplay = EGL14.eglGetCurrentDisplay();
    android.opengl.EGLSurface previousDrawSurface = EGL14.eglGetCurrentSurface(EGL_DRAW);
    android.opengl.EGLSurface previousReadSurface = EGL14.eglGetCurrentSurface(EGL_READ);

    android.opengl.EGLDisplay defaultDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
    android.opengl.EGLSurface tempEglSurface = null;

    if (!previousContext.equals(context)) {
      int[] surfaceAttribs = {EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE};
      android.opengl.EGLConfig tempConfig = getThrowawayConfig(defaultDisplay);
      tempEglSurface =
          EGL14.eglCreatePbufferSurface(previousDisplay, tempConfig, surfaceAttribs, 0);
      EGL14.eglMakeCurrent(defaultDisplay, tempEglSurface, tempEglSurface, context);
    }

    EGLContext egl10Context = egl.eglGetCurrentContext();

    if (!previousContext.equals(context)) {
      EGL14.eglMakeCurrent(
          previousDisplay, previousDrawSurface, previousReadSurface, previousContext);
      EGL14.eglDestroySurface(defaultDisplay, tempEglSurface);
    }

    return egl10Context;
  }

  private android.opengl.EGLConfig getThrowawayConfig(android.opengl.EGLDisplay display) {
    int[] attribList = {
      EGL10.EGL_SURFACE_TYPE, EGL10.EGL_PBUFFER_BIT | EGL10.EGL_WINDOW_BIT, EGL10.EGL_NONE
    };
    android.opengl.EGLConfig[] configs = new android.opengl.EGLConfig[1];
    int[] numConfigs = singleIntArray;
    if (!EGL14.eglChooseConfig(display, attribList, 0, configs, 0, 1, numConfigs, 0)) {
      throw new IllegalArgumentException("eglChooseConfig failed");
    }
    if (numConfigs[0] <= 0) {
      throw new IllegalArgumentException("No configs match requested attributes");
    }
    return configs[0];
  }

  /**
   * Merges two EGL attribute lists. The second list may be null. Values in the second list override
   * those with the same key in the first list.
   */
  private int[] mergeAttribLists(int[] list1, @Nullable int[] list2) {
    if (list2 == null) {
      return list1;
    }
    HashMap<Integer, Integer> attribMap = new HashMap<>();
    for (int[] list : new int[][] {list1, list2}) {
      for (int i = 0; i < list.length / 2; i++) {
        int key = list[2 * i];
        int value = list[2 * i + 1];
        if (key == EGL10.EGL_NONE) {
          break;
        }
        attribMap.put(key, value);
      }
    }
    int[] merged = new int[attribMap.size() * 2 + 1];
    int i = 0;
    for (Map.Entry<Integer, Integer> e : attribMap.entrySet()) {
      merged[i++] = e.getKey();
      merged[i++] = e.getValue();
    }
    merged[i] = EGL10.EGL_NONE;
    return merged;
  }

  private EGLConfig getConfig(int glVersion, @Nullable int[] additionalConfigAttributes) {
    int[] baseAttribList = {
      EGL10.EGL_RED_SIZE, 8,
      EGL10.EGL_GREEN_SIZE, 8,
      EGL10.EGL_BLUE_SIZE, 8,
      EGL10.EGL_ALPHA_SIZE, 8,
      EGL10.EGL_DEPTH_SIZE, 16,
      EGL10.EGL_RENDERABLE_TYPE, glVersion == 3 ? EGL_OPENGL_ES3_BIT_KHR : EGL_OPENGL_ES2_BIT,
      EGL10.EGL_SURFACE_TYPE, EGL10.EGL_PBUFFER_BIT | EGL10.EGL_WINDOW_BIT,
      EGL10.EGL_NONE
    };
    int[] attribList = mergeAttribLists(baseAttribList, additionalConfigAttributes);
    // First count the matching configs. Note that eglChooseConfig will return configs that
    // match *or exceed* the requirements, and will put the ones that exceed first!
    int[] numConfigs = singleIntArray;
    if (!egl.eglChooseConfig(eglDisplay, attribList, null, 0, numConfigs)) {
      throw new IllegalArgumentException("eglChooseConfig failed");
    }

    if (numConfigs[0] <= 0) {
      throw new IllegalArgumentException("No configs match requested attributes");
    }

    EGLConfig[] configs = new EGLConfig[numConfigs[0]];
    if (!egl.eglChooseConfig(eglDisplay, attribList, configs, configs.length, numConfigs)) {
      throw new IllegalArgumentException("eglChooseConfig#2 failed");
    }

    // Try to find a config that matches our bit sizes exactly.
    EGLConfig bestConfig = null;
    for (EGLConfig config : configs) {
      int r = findConfigAttrib(config, EGL10.EGL_RED_SIZE, 0);
      int g = findConfigAttrib(config, EGL10.EGL_GREEN_SIZE, 0);
      int b = findConfigAttrib(config, EGL10.EGL_BLUE_SIZE, 0);
      int a = findConfigAttrib(config, EGL10.EGL_ALPHA_SIZE, 0);
      if ((r == 8) && (g == 8) && (b == 8) && (a == 8)) {
        bestConfig = config;
        break;
      }
    }
    if (bestConfig == null) {
      bestConfig = configs[0];
    }

    return bestConfig;
  }

  private void checkEglError(String msg) {
    int error;
    if ((error = egl.eglGetError()) != EGL10.EGL_SUCCESS) {
      throw new RuntimeException(msg + ": EGL error: 0x" + Integer.toHexString(error));
    }
  }

  private int findConfigAttrib(EGLConfig config, int attribute, int defaultValue) {
    if (egl.eglGetConfigAttrib(eglDisplay, config, attribute, singleIntArray)) {
      return singleIntArray[0];
    }
    return defaultValue;
  }

}
