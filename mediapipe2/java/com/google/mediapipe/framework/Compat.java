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

package com.google.mediapipe.framework;

/**
 * Utilities for compatibility with old versions of Android.
 */
public class Compat {
  /**
   * Returns the native handle to the current EGL context. Can be used as a
   * replacement for EGL14.eglGetCurrentContext().getNativeHandle() before
   * API 17.
   */
  public static native long getCurrentNativeEGLContext();

  /**
   * Returns the native handle to the current EGL surface. Can be used as a
   * replacement for EGL14.eglGetCurrentSurface().getNativeHandle() before
   * API 17.
   */
  public static native long getCurrentNativeEGLSurface(int readdraw);
}
