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
  // True when created with PacketGetter.getTextureFrameDeferredSync(). This will result in gpuWait
  // when calling getTextureName().
  private final boolean deferredSync;

  GraphTextureFrame(long nativeHandle, long timestamp) {
    this(nativeHandle, timestamp, false);
  }

  GraphTextureFrame(long nativeHandle, long timestamp, boolean deferredSync) {
    nativeBufferHandle = nativeHandle;
    // TODO: use a single JNI call to fill in all info
    textureName = nativeGetTextureName(nativeBufferHandle);
    width = nativeGetWidth(nativeBufferHandle);
    height = nativeGetHeight(nativeBufferHandle);
    this.timestamp = timestamp;
    this.deferredSync = deferredSync;
  }

  /**
   * Returns the name of the underlying OpenGL texture.
   *
   * <p>Note: if this texture has been obtained using getTextureFrameDeferredWait, a GPU wait on the
   * producer sync will be done here. That means this method should be called on the GL context that
   * will actually use the texture. Note that in this case, it is also susceptible to a race
   * condition if release() is called after the if-check for nativeBufferHandle is already passed.
   */
  @Override
  public int getTextureName() {
    // Return special texture id 0 if handle is 0 i.e. frame is already released.
    if (nativeBufferHandle == 0) {
      return 0;
    }
    // Gpu wait only if deferredSync is true, such as when this GraphTextureFrame is created using
    // PacketGetter.getTextureFrameDeferredSync().
    if (deferredSync) {
      // Note that, if a CPU wait has already been done, the sync point will have been
      // cleared and this will turn into a no-op. See GlFenceSyncPoint::Wait.
      nativeGpuWait(nativeBufferHandle);
    }
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

  private native void nativeGpuWait(long nativeHandle);
}
