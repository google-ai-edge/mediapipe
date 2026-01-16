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
import android.content.Context;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.opengl.GLES20;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Process;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCapture.OnImageSavedCallback;
import androidx.camera.core.ImageCapture.OutputFileOptions;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.glutil.EglManager;
import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.microedition.khronos.egl.EGLSurface;

/**
 * Uses CameraX APIs for camera setup and access.
 *
 * <p>{@link CameraX} connects to the camera and provides video frames.
 */
public class CameraXPreviewHelper extends CameraHelper {
  /** Listener invoked when the camera instance is available. */
  public interface OnCameraBoundListener {
    /**
     * Called after CameraX has been bound to the lifecycle and the camera instance is available.
     */
    public void onCameraBound(Camera camera);
  }

  /**
   * Provides an Executor that wraps a single-threaded Handler.
   *
   * <p>All operations involving the surface texture should happen in a single thread, and that
   * thread should not be the main thread.
   *
   * <p>The surface provider callbacks require an Executor, and the onFrameAvailable callback
   * requires a Handler. We want everything to run on the same thread, so we need an Executor that
   * is also a Handler.
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

  private static final String TAG = "CameraXPreviewHelper";

  // Target frame and view resolution size in landscape.
  private static final Size TARGET_SIZE = new Size(1280, 720);
  private static final double ASPECT_TOLERANCE = 0.25;
  private static final double ASPECT_PENALTY = 10000;
  // Number of attempts for calculating the offset between the camera's clock and MONOTONIC clock.
  private static final int CLOCK_OFFSET_CALIBRATION_ATTEMPTS = 3;

  private final SingleThreadHandlerExecutor renderExecutor =
      new SingleThreadHandlerExecutor("RenderThread", Process.THREAD_PRIORITY_DEFAULT);

  private ProcessCameraProvider cameraProvider;
  private Preview preview;
  private ImageCapture imageCapture;
  private ImageCapture.Builder imageCaptureBuilder;
  private ExecutorService imageCaptureExecutorService;
  private Camera camera;
  private int[] textures = null;

  // Size of the camera-preview frames from the camera.
  private Size frameSize;
  // Rotation of the camera-preview frames in degrees.
  private int frameRotation;
  // Checks if the image capture use case is enabled.
  private boolean isImageCaptureEnabled = false;

  @Nullable private CameraCharacteristics cameraCharacteristics = null;

  // Focal length resolved in pixels on the frame texture. If it cannot be determined, this value
  // is Float.MIN_VALUE.
  private float focalLengthPixels = Float.MIN_VALUE;

  // Timestamp source of camera. This is retrieved from
  // CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE. When CameraCharacteristics is not available
  // the source is CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN.
  private int cameraTimestampSource = CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN;

  @Nullable private OnCameraBoundListener onCameraBoundListener = null;

  private boolean isLandscapeOrientation = false;

  /**
   * Initializes the camera and sets it up for accessing frames, using the default 1280 * 720
   * preview size.
   */
  @Override
  public void startCamera(
      Activity activity, CameraFacing cameraFacing, @Nullable SurfaceTexture surfaceTexture) {
    startCamera(activity, (LifecycleOwner) activity, cameraFacing, surfaceTexture, TARGET_SIZE);
  }

  /**
   * Initializes the camera and sets it up for accessing frames.
   *
   * @param targetSize the preview size to use. If set to {@code null}, the helper will default to
   *     1280 * 720.
   */
  public void startCamera(
      Activity activity,
      CameraFacing cameraFacing,
      @Nullable SurfaceTexture surfaceTexture,
      @Nullable Size targetSize) {
    startCamera(activity, (LifecycleOwner) activity, cameraFacing, surfaceTexture, targetSize);
  }

  /**
   * Initializes the camera and sets it up for accessing frames. This constructor also enables the
   * image capture use case from {@link CameraX}.
   *
   * @param imageCaptureBuilder Builder for an {@link ImageCapture}, this builder must contain the
   *     desired configuration options for the image capture being build (e.g. target resolution).
   * @param targetSize the preview size to use. If set to {@code null}, the helper will default to
   *     1280 * 720.
   */
  public void startCamera(
      Activity activity,
      @Nonnull ImageCapture.Builder imageCaptureBuilder,
      CameraFacing cameraFacing,
      @Nullable Size targetSize) {
    this.imageCaptureBuilder = imageCaptureBuilder;
    startCamera(activity, (LifecycleOwner) activity, cameraFacing, targetSize);
  }

  /**
   * Initializes the camera and sets it up for accessing frames. This constructor also enables the
   * image capture use case from {@link CameraX}.
   *
   * @param imageCaptureBuilder Builder for an {@link ImageCapture}, this builder must contain the
   *     desired configuration options for the image capture being build (e.g. target resolution).
   * @param targetSize the preview size to use. If set to {@code null}, the helper will default to
   *     1280 * 720.
   */
  public void startCamera(
      Activity activity,
      @Nonnull ImageCapture.Builder imageCaptureBuilder,
      CameraFacing cameraFacing,
      @Nullable SurfaceTexture surfaceTexture,
      @Nullable Size targetSize) {
    this.imageCaptureBuilder = imageCaptureBuilder;
    startCamera(activity, (LifecycleOwner) activity, cameraFacing, surfaceTexture, targetSize);
  }

  /**
   * Initializes the camera and sets it up for accessing frames.
   *
   * @param targetSize a predefined constant {@link #TARGET_SIZE}. If set to {@code null}, the
   *     helper will default to 1280 * 720.
   */
  public void startCamera(
      Context context,
      LifecycleOwner lifecycleOwner,
      CameraFacing cameraFacing,
      @Nullable Size targetSize) {
    startCamera(context, lifecycleOwner, cameraFacing, null, targetSize);
  }

  /**
   * Initializes the camera and sets it up for accessing frames.
   *
   * @param targetSize a predefined constant {@link #TARGET_SIZE}. If set to {@code null}, the
   *     helper will default to 1280 * 720.
   */
  public void startCamera(
      Context context,
      LifecycleOwner lifecycleOwner,
      CameraFacing cameraFacing,
      @Nullable SurfaceTexture surfaceTexture,
      @Nullable Size targetSize) {
    Executor mainThreadExecutor = ContextCompat.getMainExecutor(context);
    ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
        ProcessCameraProvider.getInstance(context);
    final boolean isSurfaceTextureProvided = surfaceTexture != null;

    Integer selectedLensFacing =
        cameraFacing == CameraHelper.CameraFacing.FRONT
            ? CameraMetadata.LENS_FACING_FRONT
            : CameraMetadata.LENS_FACING_BACK;
    cameraCharacteristics = getCameraCharacteristics(context, selectedLensFacing);
    targetSize = getOptimalViewSize(targetSize);
    // Falls back to TARGET_SIZE if either targetSize is not set or getOptimalViewSize() can't
    // determine the optimal view size.
    if (targetSize == null) {
      targetSize = TARGET_SIZE;
    }
    // According to CameraX documentation
    // (https://developer.android.com/training/camerax/configuration#specify-resolution):
    // "Express the resolution Size in the coordinate frame after rotating the supported sizes by
    // the target rotation."
    // Transpose width and height if using portrait orientation.
    Size rotatedSize =
        isLandscapeOrientation
            ? new Size(/* width= */ targetSize.getWidth(), /* height= */ targetSize.getHeight())
            : new Size(/* width= */ targetSize.getHeight(), /* height= */ targetSize.getWidth());

    cameraProviderFuture.addListener(
        () -> {
          try {
            cameraProvider = cameraProviderFuture.get();
          } catch (Exception e) {
            if (e instanceof InterruptedException) {
              Thread.currentThread().interrupt();
            }
            Log.e(TAG, "Unable to get ProcessCameraProvider: ", e);
            return;
          }

          preview = new Preview.Builder().setTargetResolution(rotatedSize).build();

          CameraSelector cameraSelector =
              cameraFacing == CameraHelper.CameraFacing.FRONT
                  ? CameraSelector.DEFAULT_FRONT_CAMERA
                  : CameraSelector.DEFAULT_BACK_CAMERA;

          // Provide surface texture.
          preview.setSurfaceProvider(
              renderExecutor,
              request -> {
                frameSize = request.getResolution();
                Log.d(
                    TAG,
                    String.format(
                        "Received surface request for resolution %dx%d",
                        frameSize.getWidth(), frameSize.getHeight()));

                SurfaceTexture previewFrameTexture =
                    isSurfaceTextureProvided ? surfaceTexture : createSurfaceTexture();
                previewFrameTexture.setDefaultBufferSize(
                    frameSize.getWidth(), frameSize.getHeight());

                request.setTransformationInfoListener(
                    renderExecutor,
                    transformationInfo -> {
                      frameRotation = transformationInfo.getRotationDegrees();
                      updateCameraCharacteristics();

                      if (!isSurfaceTextureProvided) {
                        // Detach the SurfaceTexture from the GL context we created earlier so that
                        // the MediaPipe pipeline can attach it.
                        // Only needed if MediaPipe pipeline doesn't provide a SurfaceTexture.
                        previewFrameTexture.detachFromGLContext();
                      }

                      OnCameraStartedListener listener = onCameraStartedListener;
                      if (listener != null) {
                        ContextCompat.getMainExecutor(context)
                            .execute(() -> listener.onCameraStarted(previewFrameTexture));
                      }
                    });

                Surface surface = new Surface(previewFrameTexture);
                Log.d(TAG, "Providing surface");
                request.provideSurface(
                    surface,
                    renderExecutor,
                    result -> {
                      Log.d(TAG, "Surface request result: " + result);
                      if (textures != null) {
                        GLES20.glDeleteTextures(1, textures, 0);
                      }
                      // Per
                      // https://developer.android.com/reference/androidx/camera/core/SurfaceRequest.Result,
                      // the surface was either never used (RESULT_INVALID_SURFACE,
                      // RESULT_REQUEST_CANCELLED, RESULT_SURFACE_ALREADY_PROVIDED) or the surface
                      // was used successfully and was eventually detached
                      // (RESULT_SURFACE_USED_SUCCESSFULLY) so we can release it now to free up
                      // resources.
                      if (!isSurfaceTextureProvided) {
                        previewFrameTexture.release();
                      }
                      surface.release();
                    });
              });

          // If we pause/resume the activity, we need to unbind the earlier preview use case, given
          // the way the activity is currently structured.
          cameraProvider.unbindAll();

          // Bind use case(s) to camera.
          final Camera boundCamera;
          if (imageCaptureBuilder != null) {
            imageCapture = imageCaptureBuilder.build();
            boundCamera =
                cameraProvider.bindToLifecycle(
                    lifecycleOwner, cameraSelector, preview, imageCapture);
            imageCaptureExecutorService = Executors.newSingleThreadExecutor();
            isImageCaptureEnabled = true;
          } else {
            boundCamera = cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview);
          }
          CameraXPreviewHelper.this.camera = boundCamera;
          OnCameraBoundListener listener = onCameraBoundListener;
          if (listener != null) {
            ContextCompat.getMainExecutor(context)
                .execute(() -> listener.onCameraBound(boundCamera));
          }
        },
        mainThreadExecutor);
  }

  /**
   * Captures a new still image and saves to a file along with application specified metadata. This
   * method works when {@link CameraXPreviewHelper#startCamera(Activity, ImageCapture.Builder,
   * CameraFacing, Size)} has been called previously enabling image capture. The callback will be
   * called only once for every invocation of this method.
   *
   * @param outputFile Save location for captured image.
   * @param onImageSavedCallback Callback to be called for the newly captured image.
   */
  public void takePicture(File outputFile, OnImageSavedCallback onImageSavedCallback) {
    takePicture(outputFile, onImageSavedCallback, imageCaptureExecutorService);
  }

  /**
   * Captures a new still image and saves to a file along with application specified metadata. This
   * method works when {@link CameraXPreviewHelper#startCamera(Activity, ImageCapture.Builder,
   * CameraFacing, Size)} has been called previously enabling image capture. The callback will be
   * called only once for every invocation of this method.
   *
   * @param outputFile Save location for captured image.
   * @param onImageSavedCallback Callback to be called for the newly captured image.
   * @param executorService Executor service to handle image capture.
   */
  public void takePicture(
      File outputFile, OnImageSavedCallback onImageSavedCallback, ExecutorService executorService) {
    if (isImageCaptureEnabled) {
      OutputFileOptions outputFileOptions = new OutputFileOptions.Builder(outputFile).build();
      imageCapture.takePicture(outputFileOptions, executorService, onImageSavedCallback);
    }
  }

  @Override
  public boolean isCameraRotated() {
    return frameRotation % 180 == 90;
  }

  @Override
  public Size computeDisplaySizeFromViewSize(Size viewSize) {
    // Camera target size is computed already, so just return the capture frame size.
    return frameSize;
  }

  @Nullable
  private Size getOptimalViewSize(@Nullable Size targetSize) {
    if (targetSize == null || cameraCharacteristics == null) {
      return null;
    }
    StreamConfigurationMap map =
        cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
    Size[] outputSizes = map.getOutputSizes(SurfaceTexture.class);

    // Find the best matching size. We give a large penalty to sizes whose aspect
    // ratio is too different from the desired one. That way we choose a size with
    // an acceptable aspect ratio if available, otherwise we fall back to one that
    // is close in width.
    Size optimalSize = null;
    double targetRatio = (double) targetSize.getWidth() / targetSize.getHeight();
    Log.d(
        TAG,
        String.format(
            "Camera target size ratio: %f width: %d", targetRatio, targetSize.getWidth()));
    double minCost = Double.MAX_VALUE;
    for (Size size : outputSizes) {
      double aspectRatio = (double) size.getWidth() / size.getHeight();
      double ratioDiff = Math.abs(aspectRatio - targetRatio);
      double cost =
          (ratioDiff > ASPECT_TOLERANCE ? ASPECT_PENALTY + ratioDiff * targetSize.getHeight() : 0)
              + Math.abs(size.getWidth() - targetSize.getWidth());
      Log.d(
          TAG,
          String.format(
              "Camera size candidate width: %d height: %d ratio: %f cost: %f",
              size.getWidth(), size.getHeight(), aspectRatio, cost));
      if (cost < minCost) {
        optimalSize = size;
        minCost = cost;
      }
    }
    if (optimalSize != null) {
      Log.d(
          TAG,
          String.format(
              "Optimal camera size width: %d height: %d",
              optimalSize.getWidth(), optimalSize.getHeight()));
    }
    return optimalSize;
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

  /**
   * Sets whether the device is in landscape orientation.
   *
   * <p>Must be called before {@link #startCamera}. Portrait orientation is assumed by default.
   */
  public void setLandscapeOrientation(boolean landscapeOrientation) {
    this.isLandscapeOrientation = landscapeOrientation;
  }

  /**
   * Sets a listener that will be invoked when CameraX is bound.
   *
   * <p>The listener will be invoked on the main thread after the next call to {@link #startCamera}.
   * The {@link Camera} instance can be used to get camera info and control the camera (e.g. zoom
   * level).
   */
  public void setOnCameraBoundListener(@Nullable OnCameraBoundListener listener) {
    this.onCameraBoundListener = listener;
  }

  private void updateCameraCharacteristics() {
    if (cameraCharacteristics != null) {
      // Queries camera timestamp source. It should be one of REALTIME or UNKNOWN
      // as documented in
      // https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics.html#SENSOR_INFO_TIMESTAMP_SOURCE.
      cameraTimestampSource =
          cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
      focalLengthPixels = calculateFocalLengthInPixels();
    }
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

  private SurfaceTexture createSurfaceTexture() {
    // Create a temporary surface to make the context current.
    EglManager eglManager = new EglManager(null);
    EGLSurface tempEglSurface = eglManager.createOffscreenSurface(1, 1);
    eglManager.makeCurrent(tempEglSurface, tempEglSurface);
    textures = new int[1];
    GLES20.glGenTextures(1, textures, 0);
    SurfaceTexture previewFrameTexture = new SurfaceTexture(textures[0]);
    return previewFrameTexture;
  }

  @Nullable
  private static CameraCharacteristics getCameraCharacteristics(
      Context context, Integer lensFacing) {
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
