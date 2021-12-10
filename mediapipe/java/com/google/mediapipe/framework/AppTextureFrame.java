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
 * A {@link TextureFrame} that represents a texture produced by the application.
 *
 * <p>The {@link #waitUntilReleased()} method can be used to wait for the consumer to be done with
 * the texture before destroying or overwriting it.
 *
 * <p>With this class, your application is the producer. The consumer can be MediaPipe (if you send
 * the frame into a MediaPipe graph using {@link PacketCreator#createGpuBuffer(TextureFrame)}) or
 * your application (if you just hand it to another part of your application without going through
 * MediaPipe).
 */
public class AppTextureFrame implements TextureFrame {
  private int textureName;
  private int width;
  private int height;
  private long timestamp = Long.MIN_VALUE;
  private boolean inUse = false;
  private boolean legacyInUse = false;  // This ignores GL context sync.
  private GlSyncToken releaseSyncToken = null;

  public AppTextureFrame(int textureName, int width, int height) {
    this.textureName = textureName;
    this.width = width;
    this.height = height;
  }

  public void setTimestamp(long timestamp) {
    this.timestamp = timestamp;
  }

  @Override
  public int getTextureName() {
    return textureName;
  }

  @Override
  public int getWidth() {
    return width;
  }

  @Override
  public int getHeight() {
    return height;
  }

  @Override
  public long getTimestamp() {
    return timestamp;
  }

  /** Returns true if a call to waitUntilReleased() would block waiting for release. */
  public boolean isNotYetReleased() {
    synchronized (this) {
      return inUse && releaseSyncToken == null;
    }
  }

  /**
   * Waits until the consumer is done with the texture.
   *
   * <p>This does a CPU wait for the texture to be complete.
   * Use {@link waitUntilReleasedWithGpuSync} whenever possible.
   */
  public void waitUntilReleased() throws InterruptedException {
    synchronized (this) {
      while (inUse && releaseSyncToken == null) {
        wait();
      }
      if (releaseSyncToken != null) {
        releaseSyncToken.waitOnCpu();
        releaseSyncToken.release();
        inUse = false;
        releaseSyncToken = null;
      }
    }
  }

  /**
   * Waits until the consumer is done with the texture.
   *
   * <p>This method must be called within the application's GL context that will overwrite the
   * TextureFrame.
   */
  public void waitUntilReleasedWithGpuSync() throws InterruptedException {
    synchronized (this) {
      while (inUse && releaseSyncToken == null) {
        wait();
      }
      if (releaseSyncToken != null) {
        releaseSyncToken.waitOnGpu();
        releaseSyncToken.release();
        inUse = false;
        releaseSyncToken = null;
      }
    }
  }

  /**
   * Returns whether the texture is currently in use.
   *
   * @deprecated this ignores cross-context sync. You should use {@link waitUntilReleased} instead,
   * because cross-context sync cannot be supported efficiently using this API.
   */
  @Deprecated
  public boolean getInUse() {
    synchronized (this) {
      return legacyInUse;
    }
  }

  /**
   * Marks the texture as currently in use.
   * <p>The producer calls this before handing the texture off to the consumer.
   */
  public void setInUse() {
    synchronized (this) {
      if (releaseSyncToken != null) {
        releaseSyncToken.release();
        releaseSyncToken = null;
      }
      inUse = true;
      legacyInUse = true;
    }
  }

  /**
   * Marks the texture as no longer in use.
   * <p>The consumer calls this when it is done using the texture.
   */
  @Override
  public void release() {
    synchronized (this) {
      inUse = false;
      legacyInUse = false;
      notifyAll();
    }
  }

  /**
   * Called by MediaPipe when the texture has been released.
   *
   * <p>The sync token can be used to ensure that the GPU is done reading from the texture.
   */
  @Override
  public void release(GlSyncToken syncToken) {
    synchronized (this) {
      if (releaseSyncToken != null) {
        releaseSyncToken.release();
        releaseSyncToken = null;
      }
      releaseSyncToken = syncToken;
      // Note: we deliberately do not set inUse to false here. Clients should call
      // waitUntilReleased. See deprecation notice on getInUse.
      legacyInUse = false;
      notifyAll();
    }
  }

  @Override
  public void finalize() {
    // Note: we do not normally want to rely on finalize to dispose of native objects. In this
    // case, however, the object is normally disposed of in the wait method; the  finalize method
    // serves as a fallback in case the application simply drops the object. The token object is
    // small, so even if its destruction is delayed, it's not a huge problem.
    if (releaseSyncToken != null) {
      releaseSyncToken.release();
      releaseSyncToken = null;
    }
  }
}
