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
  private TextureFrameConsumer cameraNewFrameListener;
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
  public void setCameraNewFrameListener(TextureFrameConsumer listener) {
    cameraNewFrameListener = listener;
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
    if (cameraNewFrameListener == null) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "cameraNewFrameListener is not set.");
    }
    converter.setConsumer(cameraNewFrameListener);
    cameraHelper.setOnCameraStartedListener(
        surfaceTexture ->
            converter.setSurfaceTextureAndAttachToGLContext(surfaceTexture, width, height));
    cameraHelper.startCamera(
        activity,
        cameraFacing == CameraFacing.FRONT
            ? CameraHelper.CameraFacing.FRONT
            : CameraHelper.CameraFacing.BACK,
        /*unusedSurfaceTexture=*/ null,
        null);
  }

  /** Stops the camera input. */
  public void stop() {
    if (converter != null) {
      converter.close();
    }
  }

  /** Returns a boolean which is true if the camera is in Portrait mode, false in Landscape mode. */
  public boolean isCameraRotated() {
    return cameraHelper.isCameraRotated();
  }
}
