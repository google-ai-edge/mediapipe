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

import android.app.Activity;
import android.graphics.SurfaceTexture;
import android.util.Size;
import javax.annotation.Nullable;

/** Abstract interface for a helper class that manages camera access. */
public abstract class CameraHelper {
  /** The listener is called when camera start is complete. */
  public interface OnCameraStartedListener {
    /**
     * Called when camera start is complete and the camera-preview frames can be accessed from the
     * surfaceTexture. The surfaceTexture can be null if it is not prepared by the CameraHelper.
     */
    public void onCameraStarted(@Nullable SurfaceTexture surfaceTexture);
  }

  protected static final String TAG = "CameraHelper";

  /** Represents the direction the camera faces relative to device screen. */
  public static enum CameraFacing {
    FRONT,
    BACK
  };

  protected OnCameraStartedListener onCameraStartedListener;

  protected CameraFacing cameraFacing;

  /**
   * Initializes the camera and sets it up for accessing frames from a custom SurfaceTexture object.
   * The SurfaceTexture object can be null when it is the CameraHelper that prepares a
   * SurfaceTexture object for grabbing frames.
   */
  public abstract void startCamera(
      Activity context, CameraFacing cameraFacing, @Nullable SurfaceTexture surfaceTexture);

  /**
   * Computes the ideal size of the camera-preview display (the area that the camera-preview frames
   * get rendered onto, potentially with scaling and rotation) based on the size of the view
   * containing the display. Returns the computed display size.
   */
  public abstract Size computeDisplaySizeFromViewSize(Size viewSize);

  /** Returns a boolean which is true if the camera is in Portrait mode, false in Landscape mode. */
  public abstract boolean isCameraRotated();

  public void setOnCameraStartedListener(@Nullable OnCameraStartedListener listener) {
    onCameraStartedListener = listener;
  }
}
