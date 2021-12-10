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
import android.util.Log;
import android.util.Size;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.components.TextureFrameConsumer;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.TextureFrame;
import javax.microedition.khronos.egl.EGLContext;

/**
 * The camera component that takes the camera input and produces MediaPipe {@link TextureFrame}
 * objects.
 */
public class CameraInput {
  private static final String TAG = "CameraInput";

  /** Represents the direction the camera faces relative to device screen. */
  public static enum CameraFacing {
    FRONT,
    BACK
  };

  private final CameraXPreviewHelper cameraHelper;
  private CameraHelper.OnCameraStartedListener customOnCameraStartedListener;
  private TextureFrameConsumer newFrameListener;
  // {@link SurfaceTexture} where the camera-preview frames can be accessed.
  private SurfaceTexture frameTexture;
  private ExternalTextureConverter converter;

  /**
   * Initializes CamereInput and requests camera permissions.
   *
   * @param activity an Android {@link Activity}.
   */
  public CameraInput(Activity activity) {
    cameraHelper = new CameraXPreviewHelper();
    PermissionHelper.checkAndRequestCameraPermissions(activity);
  }

  /**
   * Sets a callback to be invoked when new frames available.
   *
   * @param listener the callback.
   */
  public void setNewFrameListener(TextureFrameConsumer listener) {
    newFrameListener = listener;
  }

  /**
   * Sets a callback to be invoked when camera start is complete.
   *
   * @param listener the callback.
   */
  public void setOnCameraStartedListener(CameraHelper.OnCameraStartedListener listener) {
    customOnCameraStartedListener = listener;
  }

  /**
   * Sets up the external texture converter and starts the camera.
   *
   * @param activity an Android {@link Activity}.
   * @param eglContext an OpenGL {@link EGLContext}.
   * @param cameraFacing the direction the camera faces relative to device screen.
   * @param width the desired width of the converted texture.
   * @param height the desired height of the converted texture.
   */
  public void start(
      Activity activity, EGLContext eglContext, CameraFacing cameraFacing, int width, int height) {
    if (!PermissionHelper.cameraPermissionsGranted(activity)) {
      return;
    }
    if (converter == null) {
      converter = new ExternalTextureConverter(eglContext, 2);
    }
    if (newFrameListener == null) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "newFrameListener is not set.");
    }
    frameTexture = converter.getSurfaceTexture();
    converter.setConsumer(newFrameListener);
    cameraHelper.setOnCameraStartedListener(
        surfaceTexture -> {
          if (width != 0 && height != 0) {
            // Sets the size of the output texture frame.
            updateOutputSize(width, height);
          }
          if (customOnCameraStartedListener != null) {
            customOnCameraStartedListener.onCameraStarted(surfaceTexture);
          }
        });
    cameraHelper.startCamera(
        activity,
        cameraFacing == CameraFacing.FRONT
            ? CameraHelper.CameraFacing.FRONT
            : CameraHelper.CameraFacing.BACK,
        /*surfaceTexture=*/ frameTexture,
        (width == 0 || height == 0) ? null : new Size(width, height));
  }

  /**
   * Sets or updates the size of the output {@link TextureFrame}. Can be invoked by {@code
   * SurfaceHolder.Callback.surfaceChanged} when the surface size is changed.
   *
   * @param width the desired width of the converted texture.
   * @param height the desired height of the converted texture.
   */
  public void updateOutputSize(int width, int height) {
    Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(new Size(width, height));
    boolean isCameraRotated = cameraHelper.isCameraRotated();
    Log.i(
        TAG,
        "Set camera output texture frame size to width="
            + displaySize.getWidth()
            + " , height="
            + displaySize.getHeight());
    // Configure the output width and height as the computed
    // display size.
    converter.setDestinationSize(
        isCameraRotated ? displaySize.getHeight() : displaySize.getWidth(),
        isCameraRotated ? displaySize.getWidth() : displaySize.getHeight());
  }

  /** Closes the camera input. */
  public void close() {
    if (converter != null) {
      converter.close();
    }
  }

  /** Returns a boolean which is true if the camera is in Portrait mode, false in Landscape mode. */
  public boolean isCameraRotated() {
    return cameraHelper.isCameraRotated();
  }
}
