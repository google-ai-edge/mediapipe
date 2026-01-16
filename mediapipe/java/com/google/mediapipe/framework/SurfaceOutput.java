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

import javax.annotation.Nullable;

/**
 * Outputs a MediaPipe video stream to an {@link android.opengl.EGLSurface}.
 *
 * <p>Should be created using {@link Graph#addEglSurfaceOutput}.
 */
public class SurfaceOutput {
  private Packet surfaceHolderPacket;
  private Graph mediapipeGraph;

  SurfaceOutput(Graph context, Packet holderPacket) {
    mediapipeGraph = context;
    surfaceHolderPacket = holderPacket;
  }

  /**
   * Sets vertical flipping of the output surface, useful for conversion between coordinate systems
   * with top-left v.s. bottom-left origins. This should be called before {@link
   * #setSurface(Object)} or {@link #setEglSurface(long)}.
   */
  public void setFlipY(boolean flip) {
    nativeSetFlipY(surfaceHolderPacket.getNativeHandle(), flip);
  }

  /**
   * Sets whether to update the presentation timestamp of the output surface.
   *
   * <p>If true, update the surface presentation timestamp from the MediaPipe packets on Android. It
   * is set to 1000 times the packet timestamp to convert from microseconds (packet) to nanoseconds
   * (surface).
   *
   * <p>This enables consumers to control the presentation time on a SurfaceView or to recover the
   * timestamp with ImageReader or SurfaceTexture.
   *
   * <p>See https://registry.khronos.org/EGL/extensions/ANDROID/EGL_ANDROID_presentation_time.txt
   * for details about the meaning of the presentation time.
   *
   * <p>See also
   *
   * <ul>
   *   <li>https://developer.android.com/reference/android/media/Image#getTimestamp()
   *   <li>https://developer.android.com/reference/android/graphics/SurfaceTexture#getTimestamp()
   * </ul>
   */
  public void setUpdatePresentationTime(boolean updateTimestamp) {
    nativeSetUpdatePresentationTime(surfaceHolderPacket.getNativeHandle(), updateTimestamp);
  }

  /**
   * Connects an Android {@link Surface} to an output.
   *
   * <p>This creates the requisite {@link EGLSurface} internally. If one has already been created
   * for this Surface outside of MediaPipe, the call will fail.
   *
   * <p>Note that a given Surface can only be connected to one output. If you wish to move it to a
   * different output, first call {@code setSurface(null)} on the old output.
   *
   * @param surface The surface to connect. Can be {@code null}.
   */
  public void setSurface(@Nullable Object surface) {
    nativeSetSurface(
        mediapipeGraph.getNativeHandle(), surfaceHolderPacket.getNativeHandle(), surface);
  }

  /**
   * Connects an EGL surface to an output.
   *
   * <p>NOTE: The surface needs to be compatible with the GL context used by MediaPipe. In practice
   * this means the EGL context that created the surface should use the same config as used by the
   * MediaPipe GL context, otherwise the surface sink calculator will fail with {@code
   * EGL_BAD_MATCH}.
   *
   * @param nativeEglSurface Native handle to the egl surface.
   */
  public void setEglSurface(long nativeEglSurface) {
    nativeSetEglSurface(
        mediapipeGraph.getNativeHandle(), surfaceHolderPacket.getNativeHandle(), nativeEglSurface);
  }

  private native void nativeSetFlipY(long nativePacket, boolean flip);

  private native void nativeSetUpdatePresentationTime(long nativePacket, boolean updateTimestamp);

  private native void nativeSetSurface(long nativeContext, long nativePacket, Object surface);

  private native void nativeSetEglSurface(
      long nativeContext, long nativePacket, long nativeEglSurface);
}
