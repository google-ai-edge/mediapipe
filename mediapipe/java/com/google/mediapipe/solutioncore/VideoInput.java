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

import android.app.Activity;
import android.graphics.SurfaceTexture;
import android.media.MediaPlayer;
import android.net.Uri;
import android.opengl.GLES20;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Process;
import android.util.Log;
import android.view.Surface;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.components.TextureFrameConsumer;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.glutil.EglManager;
import java.io.IOException;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLSurface;

/**
 * The video player component that reads video frames from the video uri and produces MediaPipe
 * {@link TextureFrame} objects.
 */
public class VideoInput {
  /**
   * Provides an Executor that wraps a single-threaded Handler.
   *
   * <p>{@link MediaPlayer} is not thread-safe. Reference:
   * https://developer.android.com/reference/android/media/MediaPlayer. Creation of and all access
   * to the {@link MediaPlayer} instance will be be on the same thread via
   * SingleThreadHandlerExecutor.
   */
  private static final class SingleThreadHandlerExecutor implements Executor {
    private final HandlerThread handlerThread;
    private final Handler handler;

    SingleThreadHandlerExecutor(String threadName, int priority) {
      handlerThread = new HandlerThread(threadName, priority);
      handlerThread.start();
      handler = new Handler(handlerThread.getLooper());
    }

    @Override
    public void execute(Runnable command) {
      if (!handler.post(command)) {
        throw new RejectedExecutionException(handlerThread.getName() + " is shutting down.");
      }
    }

    boolean shutdown() {
      return handlerThread.quitSafely();
    }
  }

  /**
   * The state of the MediaPlayer. See
   * https://developer.android.com/reference/android/media/MediaPlayer#StateDiagram
   */
  private enum MediaPlayerState {
    IDLE,
    PREPARING,
    PREPARED,
    STARTED,
    PAUSED,
    STOPPED,
    PLAYBACK_COMPLETE,
    END,
  }

  private static final String TAG = "VideoInput";
  private final SingleThreadHandlerExecutor executor;
  private TextureFrameConsumer newFrameListener;
  private MediaPlayer mediaPlayer;
  private MediaPlayerState state = MediaPlayerState.IDLE;
  private boolean looping = false;
  private float audioVolume = 1.0f;
  // {@link SurfaceTexture} where the video frames can be accessed.
  private SurfaceTexture surfaceTexture;
  private int textureId;
  private EglManager eglManager;
  private ExternalTextureConverter converter;

  /**
   * Initializes VideoInput and requests read external storage permissions.
   *
   * @param activity an Android {@link Activity}.
   */
  public VideoInput(Activity activity) {
    PermissionHelper.checkAndRequestReadExternalStoragePermissions(activity);
    executor =
        new SingleThreadHandlerExecutor("MediaPlayerThread", Process.THREAD_PRIORITY_DEFAULT);
  }

  /**
   * Sets a callback to be invoked when new frames available.
   *
   * @param listener the MediaPipe {@link TextureFrameConsumer} callback.
   */
  public void setNewFrameListener(TextureFrameConsumer listener) {
    newFrameListener = listener;
  }

  /**
   * Sets the player to be looping or non-looping.
   *
   * @param looping whether to loop or not.
   */
  public void setLooping(boolean looping) {
    this.looping = looping;
  }

  /**
   * Sets the audio volumn.
   *
   * @param audioVolume the volumn scalar.
   */
  public void setVolume(float audioVolume) {
    this.audioVolume = audioVolume;
  }

  /**
   * Sets up the the {@link MediaPlayer} and the {@link ExternalTextureConverter}, and starts the
   * playback.
   *
   * @param activity an Android {@link Activity}.
   * @param videoUri an Android {@link Uri} to locate a local video file.
   * @param sharedContext an OpenGL {@link EGLContext}.
   * @param displayWidth the width of the display.
   * @param displayHeight the height of the display.
   */
  public void start(
      Activity activity,
      Uri videoUri,
      EGLContext sharedContext,
      int displayWidth,
      int displayHeight) {
    if (!PermissionHelper.readExternalStoragePermissionsGranted(activity)) {
      return;
    }
    if (newFrameListener == null) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "newFrameListener is not set.");
    }
    eglManager = new EglManager(sharedContext);
    createSurfaceTexture();
    converter = new ExternalTextureConverter(sharedContext, 2);
    converter.setConsumer(newFrameListener);
    executor.execute(
        () -> {
          if (state != MediaPlayerState.IDLE && state != MediaPlayerState.END) {
            return;
          }
          mediaPlayer = new MediaPlayer();
          mediaPlayer.setLooping(looping);
          mediaPlayer.setVolume(audioVolume, audioVolume);
          mediaPlayer.setOnPreparedListener(
              unused -> {
                surfaceTexture.setDefaultBufferSize(
                    mediaPlayer.getVideoWidth(), mediaPlayer.getVideoHeight());
                // Calculates the optimal texture size by preserving the video aspect ratio.
                float videoAspectRatio =
                    (float) mediaPlayer.getVideoWidth() / mediaPlayer.getVideoHeight();
                float displayAspectRatio = (float) displayWidth / displayHeight;
                int textureWidth =
                    displayAspectRatio > videoAspectRatio
                        ? (int) (displayHeight * videoAspectRatio)
                        : displayWidth;
                int textureHeight =
                    displayAspectRatio > videoAspectRatio
                        ? displayHeight
                        : (int) (displayWidth / videoAspectRatio);
                converter.setSurfaceTexture(surfaceTexture, textureWidth, textureHeight);
                state = MediaPlayerState.PREPARED;
                executor.execute(
                    () -> {
                      if (mediaPlayer != null && state == MediaPlayerState.PREPARED) {
                        mediaPlayer.start();
                        state = MediaPlayerState.STARTED;
                      }
                    });
              });
          mediaPlayer.setOnErrorListener(
              (unused, what, extra) -> {
                Log.e(
                    TAG,
                    String.format(
                        "Error during mediaPlayer initialization. what: %s extra: %s",
                        what, extra));
                executor.execute(this::close);
                return true;
              });
          mediaPlayer.setOnCompletionListener(
              unused -> {
                state = MediaPlayerState.PLAYBACK_COMPLETE;
                executor.execute(this::close);
              });
          try {
            mediaPlayer.setDataSource(activity, videoUri);
            mediaPlayer.setSurface(new Surface(surfaceTexture));
            state = MediaPlayerState.PREPARING;
            mediaPlayer.prepareAsync();
          } catch (IOException e) {
            Log.e(TAG, "Failed to start MediaPlayer:", e);
            throw new RuntimeException(e);
          }
        });
  }

  /** Pauses the playback. */
  public void pause() {
    executor.execute(
        () -> {
          if (mediaPlayer != null
              && (state == MediaPlayerState.STARTED || state == MediaPlayerState.PAUSED)) {
            mediaPlayer.pause();
            state = MediaPlayerState.PAUSED;
          }
        });
  }

  /** Resumes the paused playback. */
  public void resume() {
    executor.execute(
        () -> {
          if (mediaPlayer != null && state == MediaPlayerState.PAUSED) {
            mediaPlayer.start();
            state = MediaPlayerState.STARTED;
          }
        });
  }

  /** Stops the playback. */
  public void stop() {
    executor.execute(
        () -> {
          if (mediaPlayer != null
              && (state == MediaPlayerState.PREPARED
                  || state == MediaPlayerState.STARTED
                  || state == MediaPlayerState.PAUSED
                  || state == MediaPlayerState.PLAYBACK_COMPLETE
                  || state == MediaPlayerState.STOPPED)) {
            mediaPlayer.stop();
            state = MediaPlayerState.STOPPED;
          }
        });
  }

  /** Closes VideoInput and releases the resources. */
  public void close() {
    if (converter != null) {
      converter.close();
      converter = null;
    }
    executor.execute(
        () -> {
          if (mediaPlayer != null) {
            mediaPlayer.release();
            state = MediaPlayerState.END;
          }
          if (eglManager != null) {
            destorySurfaceTexture();
            eglManager.release();
            eglManager = null;
          }
        });
    looping = false;
    audioVolume = 1.0f;
  }

  private void createSurfaceTexture() {
    if (eglManager == null) {
      return;
    }
    // Creates a temporary surface to make the context current.
    EGLSurface tempEglSurface = eglManager.createOffscreenSurface(1, 1);
    eglManager.makeCurrent(tempEglSurface, tempEglSurface);
    int[] textures = new int[1];
    GLES20.glGenTextures(1, textures, 0);
    textureId = textures[0];
    surfaceTexture = new SurfaceTexture(textureId);
    eglManager.makeNothingCurrent();
    eglManager.releaseSurface(tempEglSurface);
  }

  private void destorySurfaceTexture() {
    if (eglManager == null) {
      return;
    }
    // Creates a temporary surface to make the context current.
    EGLSurface tempEglSurface = eglManager.createOffscreenSurface(1, 1);
    eglManager.makeCurrent(tempEglSurface, tempEglSurface);
    surfaceTexture.release();
    GLES20.glDeleteTextures(1, new int[] {textureId}, 0);
    eglManager.makeNothingCurrent();
    eglManager.releaseSurface(tempEglSurface);
    surfaceTexture = null;
  }
}
