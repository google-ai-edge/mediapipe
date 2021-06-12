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
 * A {@link TextureFrame} that represents a texture produced by MediaPipe.
 *
 * <p>The consumer is typically your application, which should therefore call the {@link #release()}
 * method.
 */
public class GraphTextureFrame implements TextureFrame {
  private long nativeBufferHandle;
  // We cache these to be able to get them without a JNI call.
  private int textureName;
  private int width;
  private int height;
  private long timestamp = Long.MIN_VALUE;

  GraphTextureFrame(long nativeHandle, long timestamp) {
    nativeBufferHandle = nativeHandle;
    // TODO: use a single JNI call to fill in all info
    textureName = nativeGetTextureName(nativeBufferHandle);
    width = nativeGetWidth(nativeBufferHandle);
    height = nativeGetHeight(nativeBufferHandle);
    this.timestamp = timestamp;
  }

  /** Returns the name of the underlying OpenGL texture. */
  @Override
  public int getTextureName() {
    return textureName;
  }

  /** Returns the width of the underlying OpenGL texture. */
  @Override
  public int getWidth() {
    return width;
  }

  /** Returns the height of the underlying OpenGL texture. */
  @Override
  public int getHeight() {
    return height;
  }

  @Override
  public long getTimestamp() {
    return timestamp;
  }

  /**
   * Releases a reference to the underlying buffer.
   *
   * <p>The consumer calls this when it is done using the texture.
   */
  @Override
  public void release() {
    if (nativeBufferHandle != 0) {
      nativeReleaseBuffer(nativeBufferHandle);
      nativeBufferHandle = 0;
    }
  }

  /**
   * Releases a reference to the underlying buffer.
   *
   * <p>This form of the method is called when the consumer is MediaPipe itself. This can occur if a
   * packet coming out of the graph is sent back into an input stream. Since both the producer and
   * the consumer use the same context, we do not need to do further synchronization. Note: we do
   * not currently support GPU sync across multiple graphs. TODO: Application consumers
   * currently cannot create a GlSyncToken, so they cannot call this method.
   */
  @Override
  public void release(GlSyncToken syncToken) {
    syncToken.release();
    release();
  }

  private native void nativeReleaseBuffer(long nativeHandle);
  private native int nativeGetTextureName(long nativeHandle);
  private native int nativeGetWidth(long nativeHandle);
  private native int nativeGetHeight(long nativeHandle);
}
