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
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import androidx.camera.core.CameraX;
import androidx.camera.core.CameraX.LensFacing;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Uses CameraX APIs for camera setup and access.
 *
 * <p>{@link CameraX} connects to the camera and provides video frames.
 */
public class CameraXPreviewHelper extends CameraHelper {
  private static final String TAG = "CameraXPreviewHelper";

  // Target frame and view resolution size in landscape.
  private static final Size TARGET_SIZE = new Size(1280, 720);

  // Number of attempts for calculating the offset between the camera's clock and MONOTONIC clock.
  private static final int CLOCK_OFFSET_CALIBRATION_ATTEMPTS = 3;

  private Preview preview;

  // Size of the camera-preview frames from the camera.
  private Size frameSize;
  // Rotation of the camera-preview frames in degrees.
  private int frameRotation;

  @Nullable private CameraCharacteristics cameraCharacteristics = null;

  // Focal length resolved in pixels on the frame texture. If it cannot be determined, this value
  // is Float.MIN_VALUE.
  private float focalLengthPixels = Float.MIN_VALUE;

  // Timestamp source of camera. This is retrieved from
  // CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE. When CameraCharacteristics is not available
  // the source is CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN.
  private int cameraTimestampSource = CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN;

  @Override
  public void startCamera(
      Activity context, CameraFacing cameraFacing, SurfaceTexture surfaceTexture) {
    startCamera(context, cameraFacing, surfaceTexture, TARGET_SIZE);
  }

  public void startCamera(
      Activity context, CameraFacing cameraFacing, SurfaceTexture surfaceTexture, Size targetSize) {
    LensFacing cameraLensFacing =
        cameraFacing == CameraHelper.CameraFacing.FRONT ? LensFacing.FRONT : LensFacing.BACK;
    PreviewConfig previewConfig =
        new PreviewConfig.Builder()
            .setLensFacing(cameraLensFacing)
            .setTargetResolution(targetSize)
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
          cameraCharacteristics = getCameraCharacteristics(context, selectedLensFacing);
          if (cameraCharacteristics != null) {
            // Queries camera timestamp source. It should be one of REALTIME or UNKNOWN as
            // documented in
            // https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics.html#SENSOR_INFO_TIMESTAMP_SOURCE.
            cameraTimestampSource =
                cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
            focalLengthPixels = calculateFocalLengthInPixels();
          }

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

  @Nullable
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

  // Computes the difference between the camera's clock and MONOTONIC clock using camera's
  // timestamp source information. This function assumes by default that the camera timestamp
  // source is aligned to CLOCK_MONOTONIC. This is useful when the camera is being used
  // synchronously with other sensors that yield timestamps in the MONOTONIC timebase, such as
  // AudioRecord for audio data. The offset is returned in nanoseconds.
  public long getTimeOffsetToMonoClockNanos() {
    if (cameraTimestampSource == CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) {
      // This clock shares the same timebase as SystemClock.elapsedRealtimeNanos(), see
      // https://developer.android.com/reference/android/hardware/camera2/CameraMetadata.html#SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME.
      return getOffsetFromRealtimeTimestampSource();
    } else {
      return getOffsetFromUnknownTimestampSource();
    }
  }

  private static long getOffsetFromUnknownTimestampSource() {
    // Implementation-wise, this timestamp source has the same timebase as CLOCK_MONOTONIC, see
    // https://stackoverflow.com/questions/38585761/what-is-the-timebase-of-the-timestamp-of-cameradevice.
    return 0L;
  }

  private static long getOffsetFromRealtimeTimestampSource() {
    // Measure the offset of the REALTIME clock w.r.t. the MONOTONIC clock. Do
    // CLOCK_OFFSET_CALIBRATION_ATTEMPTS measurements and choose the offset computed with the
    // smallest delay between measurements. When the camera returns a timestamp ts, the
    // timestamp in MONOTONIC timebase will now be (ts + cameraTimeOffsetToMonoClock).
    long offset = Long.MAX_VALUE;
    long lowestGap = Long.MAX_VALUE;
    for (int i = 0; i < CLOCK_OFFSET_CALIBRATION_ATTEMPTS; ++i) {
      long startMonoTs = System.nanoTime();
      long realTs = SystemClock.elapsedRealtimeNanos();
      long endMonoTs = System.nanoTime();
      long gapMonoTs = endMonoTs - startMonoTs;
      if (gapMonoTs < lowestGap) {
        lowestGap = gapMonoTs;
        offset = (startMonoTs + endMonoTs) / 2 - realTs;
      }
    }
    return offset;
  }

  public float getFocalLengthPixels() {
    return focalLengthPixels;
  }

  public Size getFrameSize() {
    return frameSize;
  }

  // Computes the focal length of the camera in pixels based on lens and sensor properties.
  private float calculateFocalLengthInPixels() {
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
    return frameSize.getWidth() * focalLengthMm / sensorWidthMm;
  }

  @Nullable
  private static CameraCharacteristics getCameraCharacteristics(
      Activity context, Integer lensFacing) {
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
          return availableCameraCharacteristics;
        }
      }
    } catch (CameraAccessException e) {
      Log.e(TAG, "Accessing camera ID info got error: " + e);
    }
    return null;
  }
}
