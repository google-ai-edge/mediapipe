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
import android.graphics.Bitmap;
import android.util.Log;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.TextureFrame;
import com.google.mediapipe.glutil.EglManager;
import java.util.concurrent.atomic.AtomicInteger;
import javax.microedition.khronos.egl.EGLContext;

/** The base class of the MediaPipe image solutions. */
// TODO: Consolidates the "send" methods to be a single "send(MlImage image)".
public class ImageSolutionBase extends SolutionBase {
  public static final String TAG = "ImageSolutionBase";
  protected boolean staticImageMode;
  private EglManager eglManager;
  // Internal fake timestamp for static images.
  private final AtomicInteger staticImageTimestamp = new AtomicInteger(0);

  /**
   * Initializes MediaPipe image solution base with Android context, solution specific settings, and
   * solution result handler.
   *
   * @param context an Android {@link Context}.
   * @param solutionInfo a {@link SolutionInfo} contains binary graph file path, graph input and
   *     output stream names.
   * @param outputHandler a {@link OutputHandler} handles the solution graph output packets and
   *     runtime exception.
   */
  @Override
  public synchronized void initialize(
      Context context,
      SolutionInfo solutionInfo,
      OutputHandler<? extends SolutionResult> outputHandler) {
    staticImageMode = solutionInfo.staticImageMode();
    try {
      super.initialize(context, solutionInfo, outputHandler);
      eglManager = new EglManager(/*parentContext=*/ null);
      solutionGraph.setParentGlContext(eglManager.getNativeContext());
    } catch (MediaPipeException e) {
      reportError("Error occurs while creating MediaPipe image solution graph.", e);
    }
  }

  /** Returns the managed {@link EGLContext} to share the opengl context with other components. */
  public EGLContext getGlContext() {
    return eglManager.getContext();
  }


  /** Returns the opengl major version number. */
  public int getGlMajorVersion() {
    return eglManager.getGlMajorVersion();
  }

  /** Sends a {@link TextureFrame} into solution graph for processing. */
  public void send(TextureFrame textureFrame) {
    if (!staticImageMode && textureFrame.getTimestamp() == Long.MIN_VALUE) {
      reportError(
          "Error occurs while calling the MediaPipe solution send method.",
          new MediaPipeException(
              MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
              "TextureFrame's timestamp needs to be explicitly set if not in static image mode."));
      return;
    }
    long timestampUs =
        staticImageMode ? staticImageTimestamp.getAndIncrement() : textureFrame.getTimestamp();
    sendImage(textureFrame, timestampUs);
  }

  /**
   * Sends a {@link Bitmap} with a timestamp into solution graph for processing. In static image
   * mode, the timestamp is ignored.
   */
  public void send(Bitmap inputBitmap, long timestamp) {
    if (staticImageMode) {
      Log.w(TAG, "In static image mode, the MediaPipe solution ignores the input timestamp.");
    }
    sendImage(inputBitmap, staticImageMode ? staticImageTimestamp.getAndIncrement() : timestamp);
  }

  /** Sends a {@link Bitmap} (static image) into solution graph for processing. */
  public void send(Bitmap inputBitmap) {
    if (!staticImageMode) {
      reportError(
          "Error occurs while calling the solution send method.",
          new MediaPipeException(
              MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
              "When not in static image mode, a timestamp associated with the image is required."
                  + " Use send(Bitmap inputBitmap, long timestamp) instead."));
      return;
    }
    sendImage(inputBitmap, staticImageTimestamp.getAndIncrement());
  }

  /** Internal implementation of sending Bitmap/TextureFrame into the MediaPipe solution. */
  private synchronized <T> void sendImage(T imageObj, long timestamp) {
    if (lastTimestamp >= timestamp) {
      reportError(
          "The received frame having a smaller timestamp than the processed timestamp.",
          new MediaPipeException(
              MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
              "Receving a frame with invalid timestamp."));
      return;
    }
    if (!solutionGraphStarted.get()) {
      if (imageObj instanceof TextureFrame) {
        ((TextureFrame) imageObj).release();
      }
      reportError(
          "The solution graph hasn't been successfully started or error occurs during graph"
              + " initializaton.",
          new MediaPipeException(
              MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
              "Graph is not started."));
      return;
    }
    lastTimestamp = timestamp;
    Packet imagePacket = null;
    try {
      if (imageObj instanceof TextureFrame) {
        imagePacket = packetCreator.createImage((TextureFrame) imageObj);
        imageObj = null;
        statsLogger.recordGpuInputArrival(timestamp);
      } else if (imageObj instanceof Bitmap) {
        imagePacket = packetCreator.createRgbaImage((Bitmap) imageObj);
        statsLogger.recordCpuInputArrival(timestamp);
      } else {
        reportError(
            "The input image type is not supported.",
            new MediaPipeException(
                MediaPipeException.StatusCode.UNIMPLEMENTED.ordinal(),
                "The input image type is not supported."));
      }
      try {
        // addConsumablePacketToInputStream allows the graph to take exclusive ownership of the
        // packet, which may allow for more memory optimizations.
        solutionGraph.addConsumablePacketToInputStream(
            imageInputStreamName, imagePacket, timestamp);
        // If addConsumablePacket succeeded, we don't need to release the packet ourselves.
        imagePacket = null;
      } catch (MediaPipeException e) {
        // TODO: do not suppress exceptions here!
        if (errorListener == null) {
          Log.e(TAG, "Mediapipe error: ", e);
        } else {
          throw e;
        }
      }
    } catch (RuntimeException e) {
      if (errorListener != null) {
        errorListener.onError("MediaPipe packet creation error: " + e.getMessage(), e);
      } else {
        throw e;
      }
    } finally {
      if (imagePacket != null) {
        // In case of error, addConsumablePacketToInputStream will not release the packet, so we
        // have to release it ourselves. (We could also re-try adding, but we don't).
        imagePacket.release();
      }
      if (imageObj instanceof TextureFrame) {
        if (imageObj != null) {
          // imagePacket will release frame if it has been created, but if not, we need to
          // release it.
          ((TextureFrame) imageObj).release();
        }
      }
    }
  }
}
