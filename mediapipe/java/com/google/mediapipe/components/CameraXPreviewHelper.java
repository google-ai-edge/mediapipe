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
import androidx.lifecycle.LifecycleOwner;
import android.content.Context;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.util.Log;
import android.util.Size;
import androidx.camera.core.CameraX;
import androidx.camera.core.CameraX.LensFacing;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import java.util.Arrays;
import java.util.List;

/**
 * Uses CameraX APIs for camera setup and access.
 *
 * <p>{@link CameraX} connects to the camera and provides video frames.
 */
public class CameraXPreviewHelper extends CameraHelper {
  private static final String TAG = "CameraXPreviewHelper";

  // Target frame and view resolution size in landscape.
  private static final Size TARGET_SIZE = new Size(1280, 720);

  private Preview preview;

  // Size of the camera-preview frames from the camera.
  private Size frameSize;
  // Rotation of the camera-preview frames in degrees.
  private int frameRotation;

  // Focal length resolved in pixels on the frame texture.
  private float focalLengthPixels;
  private CameraCharacteristics cameraCharacteristics = null;

  @Override
  @SuppressWarnings("RestrictTo") // See b/132705545.
  public void startCamera(
      Activity context, CameraFacing cameraFacing, SurfaceTexture surfaceTexture) {
    LensFacing cameraLensFacing =
        cameraFacing == CameraHelper.CameraFacing.FRONT ? LensFacing.FRONT : LensFacing.BACK;
    PreviewConfig previewConfig =
        new PreviewConfig.Builder()
            .setLensFacing(cameraLensFacing)
            .setTargetResolution(TARGET_SIZE)
            .build();
    preview = new Preview(previewConfig);

    preview.setOnPreviewOutputUpdateListener(
        previewOutput -> {
          if (!previewOutput.getTextureSize().equals(frameSize)) {
            frameSize = previewOutput.getTextureSize();
            frameRotation = previewOutput.getRotationDegrees();
            if (frameSize.getWidth() == 0 || frameSize.getHeight() == 0) {
              // Invalid frame size. Wait for valid input dimensions before updating display size.
              Log.d(TAG, "Invalid frameSize.");
              return;
            }
          }
          Integer selectedLensFacing =
              cameraFacing == CameraHelper.CameraFacing.FRONT
                  ? CameraMetadata.LENS_FACING_FRONT
                  : CameraMetadata.LENS_FACING_BACK;
          calculateFocalLength(context, selectedLensFacing);
          if (onCameraStartedListener != null) {
            onCameraStartedListener.onCameraStarted(previewOutput.getSurfaceTexture());
          }
        });
    CameraX.bindToLifecycle(/*lifecycleOwner=*/ (LifecycleOwner) context, preview);

  }

  @Override
  public boolean isCameraRotated() {
    return frameRotation % 180 == 90;
  }

  @Override
  public Size computeDisplaySizeFromViewSize(Size viewSize) {
    if (viewSize == null || frameSize == null) {
      // Wait for all inputs before setting display size.
      Log.d(TAG, "viewSize or frameSize is null.");
      return null;
    }

    Size optimalSize = getOptimalViewSize(viewSize);
    return optimalSize != null ? optimalSize : frameSize;
  }

  private Size getOptimalViewSize(Size targetSize) {
    if (cameraCharacteristics != null) {
      StreamConfigurationMap map =
          cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
      Size[] outputSizes = map.getOutputSizes(SurfaceTexture.class);

      int selectedWidth = -1;
      int selectedHeight = -1;
      float selectedAspectRatioDifference = 1e3f;
      float targetAspectRatio = targetSize.getWidth() / (float) targetSize.getHeight();

      // Find the smallest size >= target size with the closest aspect ratio.
      for (Size size : outputSizes) {
        float aspectRatio = (float) size.getWidth() / size.getHeight();
        float aspectRatioDifference = Math.abs(aspectRatio - targetAspectRatio);
        if (aspectRatioDifference <= selectedAspectRatioDifference) {
          if ((selectedWidth == -1 && selectedHeight == -1)
              || (size.getWidth() <= selectedWidth
                  && size.getWidth() >= frameSize.getWidth()
                  && size.getHeight() <= selectedHeight
                  && size.getHeight() >= frameSize.getHeight())) {
            selectedWidth = size.getWidth();
            selectedHeight = size.getHeight();
            selectedAspectRatioDifference = aspectRatioDifference;
          }
        }
      }
      if (selectedWidth != -1 && selectedHeight != -1) {
        return new Size(selectedWidth, selectedHeight);
      }
    }
    return null;
  }

  public float getFocalLengthPixels() {
    return focalLengthPixels;
  }

  private void calculateFocalLength(Activity context, Integer lensFacing) {
    CameraManager cameraManager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
    try {
      List<String> cameraList = Arrays.asList(cameraManager.getCameraIdList());
      for (String availableCameraId : cameraList) {
        CameraCharacteristics availableCameraCharacteristics =
            cameraManager.getCameraCharacteristics(availableCameraId);
        Integer availableLensFacing =
            availableCameraCharacteristics.get(CameraCharacteristics.LENS_FACING);
        if (availableLensFacing == null) {
          continue;
        }
        if (availableLensFacing.equals(lensFacing)) {
          cameraCharacteristics = availableCameraCharacteristics;
          break;
        }
      }
      // Focal length of the camera in millimeters.
      // Note that CameraCharacteristics returns a list of focal lengths and there could be more
      // than one focal length available if optical zoom is enabled or there are multiple physical
      // cameras in the logical camera referenced here. A theoretically correct of doing this would
      // be to use the focal length set explicitly via Camera2 API, as documented in
      // https://developer.android.com/reference/android/hardware/camera2/CaptureRequest#LENS_FOCAL_LENGTH.
      float focalLengthMm =
          cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)[0];
      // Sensor Width of the camera in millimeters.
      float sensorWidthMm =
          cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE).getWidth();
      focalLengthPixels = frameSize.getWidth() * focalLengthMm / sensorWidthMm;
    } catch (CameraAccessException e) {
      Log.e(TAG, "Accessing camera ID info got error: " + e);
    }
  }
}
