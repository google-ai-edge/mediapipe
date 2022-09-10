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

package com.google.mediapipe.components;

import static java.lang.Math.max;

import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.util.Log;
import com.google.mediapipe.framework.AppTextureFrame;
import com.google.mediapipe.framework.GlSyncToken;
import com.google.mediapipe.glutil.ExternalTextureRenderer;
import com.google.mediapipe.glutil.GlThread;
import com.google.mediapipe.glutil.ShaderUtil;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import javax.microedition.khronos.egl.EGLContext;

/**
 * Textures from {@link SurfaceTexture} are only supposed to be bound to target {@link
 * GLES11Ext#GL_TEXTURE_EXTERNAL_OES}, which is accessed using samplerExternalOES in the shader.
 * This means they cannot be used with a regular shader that expects a sampler2D. This class creates
 * a copy of the texture that can be used with {@link GLES20#GL_TEXTURE_2D} and sampler2D.
 */
public class ExternalTextureConverter implements TextureFrameProducer {
  private static final String TAG = "ExternalTextureConv"; // Max length of a tag is 23.
  private static final int DEFAULT_NUM_BUFFERS = 2; // Number of output frames allocated.
  private static final String THREAD_NAME = "ExternalTextureConverter";

  private RenderThread thread;
  private Throwable startupException = null;

  /**
   * Creates the ExternalTextureConverter to create a working copy of each camera frame.
   *
   * @param numBuffers the number of camera frames that can enter processing simultaneously.
   */
  public ExternalTextureConverter(EGLContext parentContext, int numBuffers) {
    thread = makeRenderThread(parentContext, numBuffers);
    thread.setName(THREAD_NAME);

    // Catch exceptions raised during initialization. The user has not had a chance
    // to set an exception handler yet.
    // Note: exception handling is messier than we'd like because of the way GlThread works.
    // Users of that class _should_ call waitUntilReady before using it (in particular,
    // before calling getHandler), and most do, but it's not strictly enforced. So we cannot
    // have GlThread capture exceptions and rethrow them from waitUntilReady, because that
    // method may not be called, and we don't want to hide exceptions in that case.
    // Therefore, we handle that in ExternalTextureConverter.
    final Object threadExceptionLock = new Object();
    thread.setUncaughtExceptionHandler(
        (Thread t, Throwable e) -> {
          synchronized (threadExceptionLock) {
            startupException = e;
            threadExceptionLock.notify();
          }
        });

    thread.start();
    try {
      boolean success = thread.waitUntilReady();
      if (!success) {
        // If startup failed, there must have been an exception, but our handler
        // will be called _after_ waitUntilReady returns, so wait for it.
        synchronized (threadExceptionLock) {
          while (startupException == null) {
            threadExceptionLock.wait();
          }
        }
      }
    } catch (InterruptedException ie) {
      // Someone interrupted our thread. This is not supposed to happen: we own
      // the thread, and we are not going to interrupt it. Therefore, it is not
      // reasonable for this constructor to throw an InterruptedException
      // (which is a checked exception). If it should somehow happen that the
      // thread is interrupted, let's set the interrupted flag again, log the
      // error, and throw a RuntimeException.
      Thread.currentThread().interrupt();
      Log.e(TAG, "thread was unexpectedly interrupted: " + ie.getMessage());
      throw new RuntimeException(ie);
    }

    // Rethrow initialization exception.
    thread.setUncaughtExceptionHandler(null);
    if (startupException != null) {
      thread.quitSafely();
      throw new RuntimeException(startupException);
    }
  }

  /**
   * Re-renders the current frame. Notifies all consumers as if it were a new frame. This should not
   * typically be used but can be useful for cases where the consumer has lost ownership of the most
   * recent frame and needs to get it again. This does nothing if no frame has yet been received.
   */
  public void rerenderCurrentFrame() {
    SurfaceTexture surfaceTexture = getSurfaceTexture();
    if (thread != null && surfaceTexture != null && thread.getHasReceivedFirstFrame()) {
      thread.onFrameAvailable(surfaceTexture);
    }
  }

  /**
   * Sets the new buffer pool size. This is safe to set at any time.
   *
   * This doesn't adjust the buffer pool right way. Instead, it behaves as follows:
   *
   * If the new size is smaller: Excess frames in pool are not de-allocated, but rather when frames
   * are released, they wouldn't be added back to the pool until size restriction is met.
   *
   * If the new size is greater: New frames won't be created immediately. ETC anyway creates new
   * frames when all frames in the pool are in-use, but they are only added back to the pool upon
   * release if the size allows so.
   *
   * Please note, while this property allows the buffer pool to grow temporarily if needed, there is
   * a different bufferPoolMaxSize properly that strictly enforces buffer pool doesn't grow beyond
   * size and incoming frames are dropped.
   *
   * @param bufferPoolSize the number of camera frames that can enter processing simultaneously.
   */
  public void setBufferPoolSize(int bufferPoolSize) {
    thread.setBufferPoolSize(bufferPoolSize);
  }

  /**
   * Sets the buffer pool max size. Setting to <= 0 effectively clears this property.
   *
   * If set (i.e. > 0), the value should be >= bufferPoolSize. While the API allows for setting a
   * value lower without throwing an exception, internally the higher of the 2 values is used for
   * enforcing buffer pool max size.
   *
   * When set, no TextureFrames are created beyond the specified size. New incoming
   * frames will be dropped.
   *
   * When un-set (i.e. <= 0), new TextureFrames are temporarily allocated even bufferPoolSize is
   * reached. However, they are not added back to the buffer pool upon release.
   */
  public void setBufferPoolMaxSize(int bufferPoolMaxSize) {
    thread.setBufferPoolMaxSize(bufferPoolMaxSize);
  }

  /**
   * Sets vertical flipping of the texture, useful for conversion between coordinate systems with
   * top-left v.s. bottom-left origins. This should be called before {@link
   * #setSurfaceTexture(SurfaceTexture, int, int)} or {@link
   * #setSurfaceTextureAndAttachToGLContext(SurfaceTexture, int, int)}.
   */
  public void setFlipY(boolean flip) {
    thread.setFlipY(flip);
  }

  /**
   * Sets rotation of the texture, useful for supporting landscape orientations. The value should
   * correspond to Display.getRotation(), e.g. Surface.ROTATION_0. Flipping (if any) is applied
   * before rotation. This should be called before {@link #setSurfaceTexture(SurfaceTexture, int,
   * int)} or {@link #setSurfaceTextureAndAttachToGLContext(SurfaceTexture, int, int)}.
   */
  public void setRotation(int rotation) {
    thread.setRotation(rotation);
  }

  /**
   * Sets an offset that can be used to adjust the timestamps on the camera frames, for example to
   * conform to a preferred time-base or to account for a known device latency. The offset is added
   * to each frame timetamp read by the ExternalTextureConverter.
   */
  public void setTimestampOffsetNanos(long offsetInNanos) {
    thread.setTimestampOffsetNanos(offsetInNanos);
  }

  public ExternalTextureConverter(EGLContext parentContext) {
    this(parentContext, DEFAULT_NUM_BUFFERS);
  }

  public ExternalTextureConverter(
      EGLContext parentContext, SurfaceTexture texture, int targetWidth, int targetHeight) {
    this(parentContext);
    thread.setSurfaceTexture(texture, targetWidth, targetHeight);
  }

  /**
   * Sets a callbacks to catch exceptions from our GlThread.
   *
   * <p>This can be used to catch exceptions originating from the rendering thread after
   * construction. If this is used, it is best to call it before the surface texture starts serving
   * frames, e.g. by calling it before setSurfaceTexture.
   */
  public void setUncaughtExceptionHandler(Thread.UncaughtExceptionHandler handler) {
    thread.setUncaughtExceptionHandler(handler);
  }

  /**
   * Sets the input surface texture.
   *
   * <p>The provided width and height will be the size of the converted texture, so if the input
   * surface texture is rotated (as expressed by its transformation matrix) the provided width and
   * height should be swapped.
   */
  // TODO: Clean up setSurfaceTexture methods.
  public void setSurfaceTexture(SurfaceTexture texture, int width, int height) {
    if (texture != null && (width == 0 || height == 0)) {
      throw new RuntimeException(
          "ExternalTextureConverter: setSurfaceTexture dimensions cannot be zero");
    }
    thread.getHandler().post(() -> thread.setSurfaceTexture(texture, width, height));
  }

  /**
   * Returns the input surface texture.
   *
   * <p>If setSurfaceTexture has not been called, this will be a default SurfaceTexture created by
   * this class.
   */
  public SurfaceTexture getSurfaceTexture() {
    return thread.getInternalSurfaceTexture();
  }

  /**
   * Sets width and height for the output frame.
   */
  public void setDestinationSize(int width, int height) {
    thread.setDestinationSize(width, height);
  }

  // TODO: Clean up setSurfaceTexture methods.
  public void setSurfaceTextureAndAttachToGLContext(SurfaceTexture texture, int width, int height) {
    if (texture != null && (width == 0 || height == 0)) {
      throw new RuntimeException(
          "ExternalTextureConverter: setSurfaceTexture dimensions cannot be zero");
    }
    thread
        .getHandler()
        .post(() -> thread.setSurfaceTextureAndAttachToGLContext(texture, width, height));
  }

  @Override
  public void setConsumer(TextureFrameConsumer next) {
    thread.setConsumer(next);
  }

  public void addConsumer(TextureFrameConsumer consumer) {
    thread.addConsumer(consumer);
  }

  public void removeConsumer(TextureFrameConsumer consumer) {
    thread.removeConsumer(consumer);
  }

  public void close() {
    if (thread == null) {
      return;
    }
    thread.quitSafely();
    try {
      thread.join();
    } catch (InterruptedException ie) {
      // Set the interrupted flag again, log the error, and throw a RuntimeException.
      Thread.currentThread().interrupt();
      Log.e(TAG, "thread was unexpectedly interrupted: " + ie.getMessage());
      throw new RuntimeException(ie);
    }
  }

  protected RenderThread makeRenderThread(EGLContext parentContext, int numBuffers) {
    return new RenderThread(parentContext, numBuffers);
  }

  /** The thread used to do rendering. This is only protected for testing purposes. */
  protected static class RenderThread extends GlThread
      implements SurfaceTexture.OnFrameAvailableListener {
    private static final long NANOS_PER_MICRO = 1000; // Nanoseconds in one microsecond.
    // SurfaceTexture that selected as render input
    private volatile SurfaceTexture surfaceTexture = null;
    // SurfaceTexture that created internally with current EGLContext
    private volatile SurfaceTexture internalSurfaceTexture = null;
    private int[] textures = null;
    private final List<TextureFrameConsumer> consumers;
    private volatile boolean hasReceivedFirstFrame = false;

    private final Queue<PoolTextureFrame> framesAvailable = new ArrayDeque<>();
    private int framesInUse = 0;
    private int bufferPoolSize;
    private int bufferPoolMaxSize;

    private ExternalTextureRenderer renderer = null;
    private long nextFrameTimestampOffset = 0;
    private long timestampOffsetNanos = 0;
    private long previousTimestamp = 0;
    private boolean previousTimestampValid = false;

    protected int destinationWidth = 0;
    protected int destinationHeight = 0;

    private class PoolTextureFrame extends AppTextureFrame {
      public PoolTextureFrame(int textureName, int width, int height) {
        super(textureName, width, height);
      }

      @Override
      public void release(GlSyncToken syncToken) {
        super.release(syncToken);
        poolFrameReleased(this);
      }

      @Override
      public void release() {
        super.release();
        poolFrameReleased(this);
      }
    }

    public RenderThread(EGLContext parentContext, int numBuffers) {
      super(parentContext);
      bufferPoolSize = numBuffers;
      renderer = new ExternalTextureRenderer();
      consumers = new ArrayList<>();
    }

    public void setBufferPoolSize(int bufferPoolSize) {
      this.bufferPoolSize = bufferPoolSize;
    }

    public void setBufferPoolMaxSize(int bufferPoolMaxSize) {
      this.bufferPoolMaxSize = bufferPoolMaxSize;
    }

    public void setFlipY(boolean flip) {
      renderer.setFlipY(flip);
    }

    public void setRotation(int rotation) {
      renderer.setRotation(rotation);
    }

    public void setSurfaceTexture(SurfaceTexture texture, int width, int height) {
      hasReceivedFirstFrame = false;
      if (surfaceTexture != null) {
        surfaceTexture.setOnFrameAvailableListener(null);
      }
      surfaceTexture = texture;
      if (surfaceTexture != null) {
        surfaceTexture.setOnFrameAvailableListener(this);
      }
      setDestinationSize(width, height);
    }

    public void setDestinationSize(int width, int height) {
      this.destinationWidth = width;
      this.destinationHeight = height;
    }

    public void setSurfaceTextureAndAttachToGLContext(
        SurfaceTexture texture, int width, int height) {
      setSurfaceTexture(texture, width, height);
      int[] textures = new int[1];
      GLES20.glGenTextures(1, textures, 0);
      surfaceTexture.attachToGLContext(textures[0]);
    }

    public void setConsumer(TextureFrameConsumer consumer) {
      synchronized (consumers) {
        consumers.clear();
        consumers.add(consumer);
      }
    }

    public void addConsumer(TextureFrameConsumer consumer) {
      synchronized (consumers) {
        consumers.add(consumer);
      }
    }

    public void removeConsumer(TextureFrameConsumer consumer) {
      synchronized (consumers) {
        consumers.remove(consumer);
      }
    }

    public SurfaceTexture getInternalSurfaceTexture() {
      return surfaceTexture != null ? surfaceTexture : internalSurfaceTexture;
    }

    public boolean getHasReceivedFirstFrame() {
      return hasReceivedFirstFrame;
    }

    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
      handler.post(() -> renderNext(surfaceTexture));
    }

    @Override
    public void prepareGl() {
      super.prepareGl();

      GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

      renderer.setup();

      textures = new int[1];
      GLES20.glGenTextures(1, textures, 0);
      internalSurfaceTexture = new SurfaceTexture(textures[0]);
      setSurfaceTexture(internalSurfaceTexture, 0, 0);
    }

    @Override
    public void releaseGl() {
      setSurfaceTexture(null, 0, 0);
      while (!framesAvailable.isEmpty()) {
        teardownFrame(framesAvailable.remove());
      }

      internalSurfaceTexture.release();
      if (textures != null) {
        GLES20.glDeleteTextures(1, textures, 0);
      }
      renderer.release();
      super.releaseGl(); // This releases the EGL context, so must do it after any GL calls.
    }

    public void setTimestampOffsetNanos(long offsetInNanos) {
      timestampOffsetNanos = offsetInNanos;
    }

    protected void renderNext(SurfaceTexture fromTexture) {
      if (fromTexture != surfaceTexture) {
        // Although the setSurfaceTexture and renderNext methods are correctly sequentialized on
        // the same thread, the onFrameAvailable callback is not. Therefore, it is possible for
        // onFrameAvailable to queue up a renderNext call while a setSurfaceTexture call is still
        // pending on the handler. When that happens, we should simply disregard the call.
        return;
      }
      hasReceivedFirstFrame = true;
      try {
        synchronized (consumers) {
          boolean frameUpdated = false;
          for (TextureFrameConsumer consumer : consumers) {
            AppTextureFrame outputFrame = nextOutputFrame();
            if (outputFrame == null) {
              break;
            }
            // TODO: Switch to ref-counted single copy instead of making additional
            // copies blitting to separate textures each time.
            updateOutputFrame(outputFrame);
            frameUpdated = true;
            if (consumer != null) {
              if (Log.isLoggable(TAG, Log.VERBOSE)) {
                Log.v(
                    TAG,
                    String.format(
                        "Locking tex: %d width: %d height: %d",
                        outputFrame.getTextureName(),
                        outputFrame.getWidth(),
                        outputFrame.getHeight()));
              }
              outputFrame.setInUse();
              consumer.onNewFrame(outputFrame);
            }
          }
          if (!frameUpdated) {
            // Progress the SurfaceTexture BufferQueue even if we didn't update the outputFrame,
            // which could be either because there are no consumers or bufferPoolMaxSize is reached.
            surfaceTexture.updateTexImage();
          }
        }
      } finally {
      }
    }

    private static void teardownFrame(AppTextureFrame frame) {
      GLES20.glDeleteTextures(1, new int[] {frame.getTextureName()}, 0);
    }

    private PoolTextureFrame createFrame() {
      int destinationTextureId = ShaderUtil.createRgbaTexture(destinationWidth, destinationHeight);
      Log.d(
          TAG,
          String.format(
              "Created output texture: %d width: %d height: %d",
              destinationTextureId, destinationWidth, destinationHeight));
      bindFramebuffer(destinationTextureId, destinationWidth, destinationHeight);
      return new PoolTextureFrame(destinationTextureId, destinationWidth, destinationHeight);
    }

    /**
     * Gets next available frame or creates new one if next frame is not initialized or cannot be
     * used with current surface texture.
     *
     * <ul>
     *   <li>Makes sure frame width and height are same as current surface texture
     *   <li>Makes sure frame is not in use (blocks thread until frame is released)
     * </ul>
     *
     * NOTE: must be invoked on GL thread
     */
    private AppTextureFrame nextOutputFrame() {
      PoolTextureFrame outputFrame;
      synchronized (this) {
        outputFrame = framesAvailable.poll();
        // Don't create new frame if bufferPoolMaxSize is set (i.e. > 0) and reached.
        if (outputFrame == null && bufferPoolMaxSize > 0
                && framesInUse >= max(bufferPoolMaxSize, bufferPoolSize)) {
          Log.d(TAG, "Enforcing buffer pool max Size. FramesInUse: "
                  + framesInUse + " >= " + bufferPoolMaxSize);
          return null;
        }
        framesInUse++;
      }
      if (outputFrame == null) {
        outputFrame = createFrame();
      } else if (outputFrame.getWidth() != destinationWidth
          || outputFrame.getHeight() != destinationHeight) {
        // Create anew if size has changed.
        // TODO: waiting for the consumer sync here may not be necessary.
        waitUntilReleased(outputFrame);
        teardownFrame(outputFrame);
        outputFrame = createFrame();
      } else {
        // Note: waitUntilReleased does two things: waits for the frame to be released by the CPU,
        // and syncs with the GPU sync token provided by the consumer. The first part is redundant
        // here (and completes immediately), but the second part is still needed.
        waitUntilReleased(outputFrame);
      }
      return outputFrame;
    }

    protected synchronized void poolFrameReleased(PoolTextureFrame frame) {
      framesAvailable.offer(frame);
      framesInUse--;
      int keep = max(bufferPoolSize - framesInUse, 0);
      while (framesAvailable.size() > keep) {
        PoolTextureFrame textureFrameToRemove = framesAvailable.remove();
        handler.post(() -> teardownFrame(textureFrameToRemove));
      }
    }

    /**
     * Updates output frame with current pixels of surface texture and corresponding timestamp.
     *
     * @param outputFrame {@link AppTextureFrame} to populate.
     *     <p>NOTE: must be invoked on GL thread
     */
    private void updateOutputFrame(AppTextureFrame outputFrame) {
      // Copy surface texture's pixels to output frame
      bindFramebuffer(outputFrame.getTextureName(), destinationWidth, destinationHeight);
      renderer.render(surfaceTexture);

      // Populate frame timestamp with surface texture timestamp after render() as renderer
      // ensures that surface texture has the up-to-date timestamp. (Also adjust
      // |nextFrameTimestampOffset| to ensure that timestamps increase monotonically.)
      long textureTimestamp =
          (surfaceTexture.getTimestamp() + timestampOffsetNanos) / NANOS_PER_MICRO;
      if (previousTimestampValid
          && textureTimestamp + nextFrameTimestampOffset <= previousTimestamp) {
        nextFrameTimestampOffset = previousTimestamp + 1 - textureTimestamp;
      }
      outputFrame.setTimestamp(textureTimestamp + nextFrameTimestampOffset);
      previousTimestamp = outputFrame.getTimestamp();
      previousTimestampValid = true;
    }

    private void waitUntilReleased(AppTextureFrame frame) {
      try {
        if (Log.isLoggable(TAG, Log.VERBOSE)) {
          Log.v(
              TAG,
              String.format(
                  "Waiting for tex: %d width: %d height: %d timestamp: %d",
                  frame.getTextureName(),
                  frame.getWidth(),
                  frame.getHeight(),
                  frame.getTimestamp()));
        }
        frame.waitUntilReleasedWithGpuSync();
        if (Log.isLoggable(TAG, Log.VERBOSE)) {
          Log.v(
              TAG,
              String.format(
                  "Finished waiting for tex: %d width: %d height: %d timestamp: %d",
                  frame.getTextureName(),
                  frame.getWidth(),
                  frame.getHeight(),
                  frame.getTimestamp()));
        }
      } catch (InterruptedException ie) {
        // Someone interrupted our thread. This is not supposed to happen: we own
        // the thread, and we are not going to interrupt it. If it should somehow
        // happen that the thread is interrupted, let's set the interrupted flag
        // again, log the error, and throw a RuntimeException.
        Thread.currentThread().interrupt();
        Log.e(TAG, "thread was unexpectedly interrupted: " + ie.getMessage());
        throw new RuntimeException(ie);
      }
    }
  }
}
