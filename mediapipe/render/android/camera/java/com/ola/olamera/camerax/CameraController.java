/*
 * Copyright 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ola.olamera.camerax;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.hardware.camera2.CaptureResult;
import android.hardware.display.DisplayManager;
import android.os.Build;
import android.util.Log;
import android.util.Size;
import android.view.Display;

import com.google.common.util.concurrent.ListenableFuture;
import com.ola.olamera.camerax.controller.ForwardingLiveData;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;

import androidx.annotation.FloatRange;
import androidx.annotation.IntDef;
import androidx.annotation.MainThread;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.annotation.RestrictTo;
import androidx.annotation.VisibleForTesting;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraControl;
import androidx.camera.core.CameraInfo;
import androidx.camera.core.CameraInfoUnavailableException;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.FocusMeteringAction;
import androidx.camera.core.FocusMeteringResult;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Logger;
import androidx.camera.core.MeteringPoint;
import androidx.camera.core.MeteringPointFactory;
import androidx.camera.core.Preview;
import androidx.camera.core.UseCase;
import androidx.camera.core.UseCaseGroup;
import androidx.camera.core.ViewPort;
import androidx.camera.core.ZoomState;
import androidx.camera.core.impl.CameraInfoInternal;
import androidx.camera.core.impl.CameraInternal;
import androidx.camera.core.impl.ImageOutputConfig;
import androidx.camera.core.impl.Observable;
import androidx.camera.core.impl.utils.Threads;
import androidx.camera.core.impl.utils.executor.CameraXExecutors;
import androidx.camera.core.impl.utils.futures.FutureCallback;
import androidx.camera.core.impl.utils.futures.Futures;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.util.Preconditions;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public abstract class CameraController {

    private static final String TAG = "CameraController";

    // Externally visible error messages.
    private static final String CAMERA_NOT_INITIALIZED = "Camera not initialized.";
    private static final String PREVIEW_VIEW_NOT_ATTACHED = "PreviewView not attached.";
    private static final String CAMERA_NOT_ATTACHED = "Use cases not attached to camera.";
    private static final String IMAGE_CAPTURE_DISABLED = "ImageCapture disabled.";

    // Auto focus is 1/6 of the area.
    private static final float AF_SIZE = 1.0f / 6.0f;
    private static final float AE_SIZE = AF_SIZE * 1.5f;


    @Retention(RetentionPolicy.SOURCE)
    @RestrictTo(RestrictTo.Scope.LIBRARY)
    @IntDef(value = {TAP_TO_FOCUS_NOT_STARTED, TAP_TO_FOCUS_STARTED, TAP_TO_FOCUS_SUCCESSFUL,
            TAP_TO_FOCUS_UNSUCCESSFUL, TAP_TO_FOCUS_FAILED})
    public @interface TapToFocusStates {
    }

    /**
     * No tap-to-focus action has been started by the end user.
     */
    public static final int TAP_TO_FOCUS_NOT_STARTED = 0;

    /**
     * A tap-to-focus action has started but not completed. The app also gets notified with this
     * state if a new action happens before the previous one could finish.
     */
    public static final int TAP_TO_FOCUS_STARTED = 1;

    /**
     * The previous tap-to-focus action was completed successfully and the camera is focused.
     */
    public static final int TAP_TO_FOCUS_SUCCESSFUL = 2;

    /**
     * The previous tap-to-focus action was completed successfully but the camera is still
     * unfocused. It happens when CameraX receives
     * {@link CaptureResult#CONTROL_AF_STATE_NOT_FOCUSED_LOCKED}. The end user might be able to
     * get a better result by trying again with different camera distances and/or lighting.
     */
    public static final int TAP_TO_FOCUS_UNSUCCESSFUL = 3;

    /**
     * The previous tap-to-focus action was failed to complete. This is usually due to device
     * limitations.
     */
    public static final int TAP_TO_FOCUS_FAILED = 4;

    /**
     * Bitmask options to enable/disable use cases.
     *
     * @hide
     */
    @Retention(RetentionPolicy.SOURCE)
    @RestrictTo(RestrictTo.Scope.LIBRARY)
    @IntDef(flag = true, value = {IMAGE_CAPTURE, IMAGE_ANALYSIS, VIDEO_CAPTURE})
    public @interface UseCases {
    }

    /**
     * Bitmask option to enable {@link ImageCapture}. In {@link #setEnabledUseCases}, if
     * (enabledUseCases & IMAGE_CAPTURE) != 0, then controller will enable image capture features.
     */
    public static final int IMAGE_CAPTURE = 1;
    /**
     * Bitmask option to enable {@link ImageAnalysis}. In {@link #setEnabledUseCases}, if
     * (enabledUseCases & IMAGE_ANALYSIS) != 0, then controller will enable image analysis features.
     */
    public static final int IMAGE_ANALYSIS = 1 << 1;
    /**
     * Bitmask option to enable video capture use case. In {@link #setEnabledUseCases}, if
     * (enabledUseCases & VIDEO_CAPTURE) != 0, then controller will enable video capture features.
     */
    public static final int VIDEO_CAPTURE = 1 << 2;

    CameraSelector mCameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

    // By default, ImageCapture and ImageAnalysis are enabled. VideoCapture is disabled.
    private int mEnabledUseCases = IMAGE_CAPTURE;

    // CameraController and PreviewView hold reference to each other. The 2-way link is managed
    // by PreviewView.
    // Synthetic access
    @SuppressWarnings("WeakerAccess")
    @NonNull
    Preview mPreview;

    @Nullable
    OutputSize mPreviewTargetSize;

    // Synthetic access
    @SuppressWarnings("WeakerAccess")
    @NonNull
    ImageCapture mImageCapture;

    @Nullable
    OutputSize mImageCaptureTargetSize;

    @Nullable
    Executor mImageCaptureIoExecutor;

    @NonNull
    ImageAnalysis mImageAnalysis;

    @SuppressWarnings("WeakerAccess")
    @Nullable
    Camera mCamera;

    // Synthetic access
    @SuppressWarnings("WeakerAccess")
    @Nullable
    ProcessCameraProvider mCameraProvider;

    // Synthetic access
    @SuppressWarnings("WeakerAccess")
    @Nullable
    ViewPort mViewPort;

    // Synthetic access
    @SuppressWarnings("WeakerAccess")
    @Nullable
    Preview.SurfaceProvider mSurfaceProvider;

    // Synthetic access
    @SuppressWarnings("WeakerAccess")
    @Nullable
    Display mPreviewDisplay;

    /**
     * 缩放和轻点对焦的开关
     */
    private boolean mPinchToZoomEnabled = true;
    private boolean mTapToFocusEnabled = true;

    private final ForwardingLiveData<ZoomState> mZoomState = new ForwardingLiveData<>();
    private final ForwardingLiveData<Integer> mTorchState = new ForwardingLiveData<>();
    private final MutableLiveData<CameraInfo> mCameraInfoLiveData = new MutableLiveData<>();
    private final MutableLiveData<CameraControl> mCameraControlLiveData = new MutableLiveData<>();
    private final MutableLiveData<Observable<CameraInternal.State>> mStateLiveData = new ForwardingLiveData<>();
    // Synthetic access
    @SuppressWarnings("WeakerAccess")
    final MutableLiveData<Integer> mTapToFocusState = new MutableLiveData<>(
            TAP_TO_FOCUS_NOT_STARTED);

    protected final Context mAppContext;

    @NonNull
    private final ListenableFuture<Void> mInitializationFuture;

    private static final Size CAPTURE_MAX_SIZE = new Size(3456, 4608);
    private static final Size PREVIEW_MAX_SIZE = new Size(1500, 2000);

    private static final Size CAPTURE_TARGET_SIZE = new Size(3000, 4000);


    CameraController(@NonNull Context context, Size previewView) {
        this(context, previewView, true, null, null);
    }

    @SuppressLint("RestrictedApi")
    CameraController(@NonNull Context context, Size previewSize, boolean isLimitCaptureSize, Size maxCaptureSize, @Nullable OnCameraProviderListener listener) {
        mAppContext = getApplicationContext(context);
        Preview.Builder builder = new Preview.Builder();
        if (previewSize != null) {
            builder.setTargetResolution(new Size(previewSize.getHeight(), previewSize.getWidth()));
        } else {
            builder.setTargetAspectRatio(AspectRatio.RATIO_4_3);
        }
        mPreview = builder.setMaxResolution(PREVIEW_MAX_SIZE).build();

        if (maxCaptureSize == null) {
            maxCaptureSize = CAPTURE_MAX_SIZE;
        }
        ImageCapture.Builder captureBuilder = new ImageCapture.Builder()
                .setMaxResolution(maxCaptureSize)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setBufferFormat(ImageFormat.JPEG);
        if (isLimitCaptureSize) {
            // 指定拍摄分辨率为3000 * 4000
            captureBuilder.setTargetResolution(CAPTURE_TARGET_SIZE);
        } else {
            captureBuilder.setTargetAspectRatio(AspectRatio.RATIO_4_3);
        }

        mImageCapture = captureBuilder.build();
        mImageAnalysis = new ImageAnalysis.Builder().build();

        // Wait for camera to be initialized before binding use cases.
        mInitializationFuture = Futures.transform(
                ProcessCameraProvider.getInstance(mAppContext),
                provider -> {
                    mCameraProvider = provider;
                    // 用于外层在开启相机前获取某个cameraId的配置
                    if (listener != null) {
                        listener.onProcessCameraProvider(provider);
                    }
                    startCameraAndTrackStates();
                    return null;
                }, CameraXExecutors.mainThreadExecutor());
    }

    /**
     * Gets the application context and preserves the attribution tag.
     * <p>
     * TODO(b/185272953): instrument test getting attribution tag once the view artifact depends
     * on a core version that has the fix.
     */
    private static Context getApplicationContext(@NonNull Context context) {
        Context applicationContext = context.getApplicationContext();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            String attributeTag = Api30Impl.getAttributionTag(context);

            if (attributeTag != null) {
                return Api30Impl.createAttributionContext(applicationContext, attributeTag);
            }
        }

        return applicationContext;
    }

    @NonNull
    public ListenableFuture<Void> getInitializationFuture() {
        return mInitializationFuture;
    }

    /**
     * Implemented by children to refresh after {@link UseCase} is changed.
     *
     * @return
     */
    @Nullable
    abstract Camera startCamera();

    private boolean isCameraInitialized() {
        return mCameraProvider != null;
    }

    private boolean isPreviewViewAttached() {
        return mSurfaceProvider != null && mViewPort != null && mPreviewDisplay != null;
    }

    public boolean isCameraAttached() {
        return mCamera != null;
    }

    @SuppressLint("RestrictedApi")
    @MainThread
    public void setEnabledUseCases(@UseCases int enabledUseCases) {
        Threads.checkMainThread();
        if (enabledUseCases == mEnabledUseCases) {
            return;
        }
        int oldEnabledUseCases = mEnabledUseCases;
        mEnabledUseCases = enabledUseCases;
        startCameraAndTrackStates(() -> mEnabledUseCases = oldEnabledUseCases);
    }

    /**
     * Checks if the given use case mask is enabled.
     *
     * @param useCaseMask One of the {@link #IMAGE_CAPTURE}, {@link #IMAGE_ANALYSIS} or
     *                    {@link #VIDEO_CAPTURE}
     * @return true if the use case is enabled.
     */
    private boolean isUseCaseEnabled(int useCaseMask) {
        return (mEnabledUseCases & useCaseMask) != 0;
    }

    /**
     * Sets the target aspect ratio or target resolution based on {@link OutputSize}.
     */
    @SuppressLint("RestrictedApi")
    private void setTargetOutputSize(@NonNull ImageOutputConfig.Builder<?> builder,
                                     @Nullable OutputSize outputSize) {
        if (outputSize == null) {
            return;
        }
        if (outputSize.getResolution() != null) {
            builder.setTargetResolution(outputSize.getResolution());
        } else if (outputSize.getAspectRatio() != OutputSize.UNASSIGNED_ASPECT_RATIO) {
            builder.setTargetAspectRatio(outputSize.getAspectRatio());
        } else {
            Logger.e(TAG, "Invalid target surface size. " + outputSize);
        }
    }

    /**
     * Checks if two {@link OutputSize} are equal.
     */
    private boolean isOutputSizeEqual(
            @Nullable OutputSize currentSize,
            @Nullable OutputSize newSize) {
        if (currentSize == newSize) {
            return true;
        }
        return currentSize != null && currentSize.equals(newSize);
    }

    // ------------------
    // Preview use case.
    // ------------------

    @SuppressLint({"MissingPermission", "WrongConstant", "RestrictedApi"})
    @MainThread
    public void attachPreviewSurface(@NonNull Preview.SurfaceProvider surfaceProvider,
                                     @NonNull ViewPort viewPort, @NonNull Display display) {
        Threads.checkMainThread();
        if (mSurfaceProvider != surfaceProvider) {
            mSurfaceProvider = surfaceProvider;
            mPreview.setSurfaceProvider(surfaceProvider);
        }
        mViewPort = viewPort;
        mPreviewDisplay = display;
        startCameraAndTrackStates();
    }


    @MainThread
    public void clearPreviewSurface() {
        if (mCameraProvider != null) {
            // Preview is required. Unbind everything if Preview is down.
            mCameraProvider.unbindAll();
        }
        mPreview.setSurfaceProvider(null);
        mCamera = null;
        mSurfaceProvider = null;
        mViewPort = null;
        mPreviewDisplay = null;
    }

    private DisplayManager getDisplayManager() {
        return (DisplayManager) mAppContext.getSystemService(Context.DISPLAY_SERVICE);
    }


    @SuppressLint("RestrictedApi")
    @MainThread
    public void setPreviewTargetSize(@Nullable OutputSize targetSize) {
        Threads.checkMainThread();
        if (isOutputSizeEqual(mPreviewTargetSize, targetSize)) {
            return;
        }
        mPreviewTargetSize = targetSize;
        unbindPreviewAndRecreate();
        startCameraAndTrackStates();
    }

    /**
     * Returns the intended output size for {@link Preview} set by
     * {@link #setPreviewTargetSize(OutputSize)}, or null if not set.
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    @Nullable
    public OutputSize getPreviewTargetSize() {
        Threads.checkMainThread();
        return mPreviewTargetSize;
    }

    /**
     * Unbinds {@link Preview} and recreates with the latest parameters.
     */
    private void unbindPreviewAndRecreate() {
        if (isCameraInitialized()) {
            mCameraProvider.unbind(mPreview);
        }
        Preview.Builder builder = new Preview.Builder();
        setTargetOutputSize(builder, mPreviewTargetSize);
        mPreview = builder.build();
    }

    // ----------------------
    // ImageCapture UseCase.
    // ----------------------

    /**
     * Checks if {@link ImageCapture} is enabled.
     *
     * <p> {@link ImageCapture} is enabled by default. It has to be enabled before
     * {@link #takePicture} can be called.
     *
     * @see ImageCapture
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    public boolean isImageCaptureEnabled() {
        Threads.checkMainThread();
        return isUseCaseEnabled(IMAGE_CAPTURE);
    }

    /**
     * Gets the flash mode for {@link ImageCapture}.
     *
     * @return the flashMode. Value is {@link ImageCapture#FLASH_MODE_AUTO},
     * {@link ImageCapture#FLASH_MODE_ON}, or {@link ImageCapture#FLASH_MODE_OFF}.
     * @see ImageCapture
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    @ImageCapture.FlashMode
    public int getImageCaptureFlashMode() {
        Threads.checkMainThread();
        return mImageCapture.getFlashMode();
    }

    /**
     * Sets the flash mode for {@link ImageCapture}.
     *
     * <p>If not set, the flash mode will default to {@link ImageCapture#FLASH_MODE_OFF}.
     *
     * @param flashMode the flash mode for {@link ImageCapture}.
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    public void setImageCaptureFlashMode(@ImageCapture.FlashMode int flashMode) {
        Threads.checkMainThread();
        mImageCapture.setFlashMode(flashMode);
    }

    @SuppressLint("RestrictedApi")
    @MainThread
    public void takePicture(
            @NonNull ImageCapture.OutputFileOptions outputFileOptions,
            @NonNull Executor executor,
            @NonNull ImageCapture.OnImageSavedCallback imageSavedCallback,
            Rect viewPortCropRect) {
        Threads.checkMainThread();
        Preconditions.checkState(isCameraInitialized(), CAMERA_NOT_INITIALIZED);
        Preconditions.checkState(isImageCaptureEnabled(), IMAGE_CAPTURE_DISABLED);

        mImageCapture.setViewPortCropRect(viewPortCropRect);
        takePicture(outputFileOptions, executor, imageSavedCallback);
    }


    @SuppressLint("RestrictedApi")
    @MainThread
    public void takePicture(
            @NonNull ImageCapture.OutputFileOptions outputFileOptions,
            @NonNull Executor executor,
            @NonNull ImageCapture.OnImageSavedCallback imageSavedCallback) {

        Threads.checkMainThread();
        Preconditions.checkState(isCameraInitialized(), CAMERA_NOT_INITIALIZED);
        Preconditions.checkState(isImageCaptureEnabled(), IMAGE_CAPTURE_DISABLED);

        updateMirroringFlagInOutputFileOptions(outputFileOptions);
        mImageCapture.takePicture(outputFileOptions, executor, imageSavedCallback);
    }

    /**
     * Update {@link ImageCapture.OutputFileOptions} based on config.
     *
     * <p> Mirror the output image if front camera is used and if the flag is not set explicitly by
     * the app.
     *
     * @hide
     */
    @SuppressLint("RestrictedApi")
    @VisibleForTesting
    @RestrictTo(RestrictTo.Scope.LIBRARY_GROUP)
    void updateMirroringFlagInOutputFileOptions(
            @NonNull ImageCapture.OutputFileOptions outputFileOptions) {
        if (mCameraSelector.getLensFacing() != null
                && !outputFileOptions.getMetadata().isReversedHorizontalSet()) {
            outputFileOptions.getMetadata().setReversedHorizontal(
                    mCameraSelector.getLensFacing() == CameraSelector.LENS_FACING_FRONT);
        }
    }

    /**
     * Captures a new still image for in memory access.
     *
     * <p>The listener is responsible for calling {@link ImageProxy#close()} on the returned image.
     *
     * @param executor The executor in which the callback methods will be run.
     * @param callback Callback to be invoked for the newly captured image
     * @see ImageCapture#takePicture(Executor, ImageCapture.OnImageCapturedCallback)
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    public void takePicture(
            @NonNull Executor executor,
            @NonNull ImageCapture.OnImageCapturedCallback callback) {
        Threads.checkMainThread();
        Preconditions.checkState(isCameraInitialized(), CAMERA_NOT_INITIALIZED);
        Preconditions.checkState(isImageCaptureEnabled(), IMAGE_CAPTURE_DISABLED);
        mImageCapture.takePicture(executor, callback);
    }

    /**
     * Sets the image capture mode.
     *
     * <p>Valid capture modes are {@link ImageCapture.CaptureMode#CAPTURE_MODE_MINIMIZE_LATENCY},
     * which prioritizes latency over image quality, or
     * {@link ImageCapture.CaptureMode#CAPTURE_MODE_MAXIMIZE_QUALITY},
     * which prioritizes image quality over latency.
     *
     * @param captureMode the requested image capture mode.
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    public void setImageCaptureMode(@ImageCapture.CaptureMode int captureMode) {
        Threads.checkMainThread();
        if (mImageCapture.getCaptureMode() == captureMode) {
            return;
        }
        unbindImageCaptureAndRecreate(captureMode);
        startCameraAndTrackStates();
    }

    /**
     * Returns the image capture mode.
     *
     * @see ImageCapture#getCaptureMode()
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    public int getImageCaptureMode() {
        Threads.checkMainThread();
        return mImageCapture.getCaptureMode();
    }

    @SuppressLint("RestrictedApi")
    @MainThread
    public void setImageCaptureTargetSize(@Nullable OutputSize targetSize) {
        Threads.checkMainThread();
        if (isOutputSizeEqual(mImageCaptureTargetSize, targetSize)) {
            return;
        }
        mImageCaptureTargetSize = targetSize;
        unbindImageCaptureAndRecreate(getImageCaptureMode());
        startCameraAndTrackStates();
    }

    /**
     * Returns the intended output size for {@link ImageCapture} set by
     * {@link #setImageCaptureTargetSize(OutputSize)}, or null if not set.
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    @Nullable
    public OutputSize getImageCaptureTargetSize() {
        Threads.checkMainThread();
        return mImageCaptureTargetSize;
    }

    /**
     * Sets the default executor that will be used for {@link ImageCapture} IO tasks.
     *
     * <p> This executor will be used for any IO tasks specifically for {@link ImageCapture},
     * such as {@link #takePicture(ImageCapture.OutputFileOptions, Executor,
     * ImageCapture.OnImageSavedCallback)}. If no executor is set, then a default Executor
     * specifically for IO will be used instead.
     *
     * @param executor The executor which will be used for IO tasks.
     *                 TODO(b/187842789) add @see link for ImageCapture.
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    public void setImageCaptureIoExecutor(@Nullable Executor executor) {
        Threads.checkMainThread();
        if (mImageCaptureIoExecutor == executor) {
            return;
        }
        mImageCaptureIoExecutor = executor;
        unbindImageCaptureAndRecreate(mImageCapture.getCaptureMode());
        startCameraAndTrackStates();
    }


    /**
     * Unbinds {@link ImageCapture} and recreates with the latest parameters.
     */
    private void unbindImageCaptureAndRecreate(int imageCaptureMode) {
        if (isCameraInitialized()) {
            mCameraProvider.unbind(mImageCapture);
        }
        ImageCapture.Builder builder = new ImageCapture.Builder().setCaptureMode(imageCaptureMode);
        setTargetOutputSize(builder, mImageCaptureTargetSize);
        if (mImageCaptureIoExecutor != null) {
            builder.setIoExecutor(mImageCaptureIoExecutor);
        }
        mImageCapture = builder.build();
    }

    @SuppressLint("RestrictedApi")
    @MainThread
    public boolean isImageAnalysisEnabled() {
        Threads.checkMainThread();
        return isUseCaseEnabled(IMAGE_ANALYSIS);
    }

    // -----------------
    // Camera control
    // -----------------

    @SuppressLint("RestrictedApi")
    @MainThread
    public void initCameraSelector(@NonNull CameraSelector cameraSelector) {
        Threads.checkMainThread();
        if (mCameraSelector == cameraSelector) {
            return;
        }
        mCameraSelector = cameraSelector;
    }

    @SuppressLint("RestrictedApi")
    @MainThread
    public void setCameraSelector(@NonNull CameraSelector cameraSelector) {
        Threads.checkMainThread();
        if (mCameraSelector == cameraSelector) {
            return;
        }

        CameraSelector oldCameraSelector = mCameraSelector;
        mCameraSelector = cameraSelector;

        if (mCameraProvider == null) {
            return;
        }
        mCameraProvider.unbindAll();
        startCameraAndTrackStates(() -> mCameraSelector = oldCameraSelector);
    }

    /**
     * Checks if the given {@link CameraSelector} can be resolved to a camera.
     *
     * <p> Use this method to check if the device has the given camera.
     *
     * <p> Only call this method after camera is initialized. e.g. after the
     * {@link ListenableFuture} from {@link #getInitializationFuture()} is finished. Calling it
     * prematurely throws {@link IllegalStateException}. Example:
     *
     * <pre><code>
     * controller.getInitializationFuture().addListener(() -> {
     *     if (controller.hasCamera(cameraSelector)) {
     *         controller.setCameraSelector(cameraSelector);
     *     } else {
     *         // Update UI if the camera is not available.
     *     }
     *     // Attach PreviewView after we know the camera is available.
     *     previewView.setController(controller);
     * }, ContextCompat.getMainExecutor(requireContext()));
     * </code></pre>
     *
     * @return true if the {@link CameraSelector} can be resolved to a camera.
     * @throws IllegalStateException if the camera is not initialized.
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    public boolean hasCamera(@NonNull CameraSelector cameraSelector) {
        Threads.checkMainThread();
        Preconditions.checkNotNull(cameraSelector);

        if (mCameraProvider == null) {
            throw new IllegalStateException("Camera not initialized. Please wait for "
                    + "the initialization future to finish. See #getInitializationFuture().");
        }

        try {
            return mCameraProvider.hasCamera(cameraSelector);
        } catch (CameraInfoUnavailableException e) {
            Logger.w(TAG, "Failed to check camera availability", e);
            return false;
        }
    }

    /**
     * Gets the {@link CameraSelector}.
     *
     * <p>The default value is{@link CameraSelector#DEFAULT_BACK_CAMERA}.
     *
     * @see CameraSelector
     */
    @SuppressLint("RestrictedApi")
    @NonNull
    @MainThread
    public CameraSelector getCameraSelector() {
        Threads.checkMainThread();
        return mCameraSelector;
    }

    /**
     * Returns whether pinch-to-zoom is enabled.
     *
     * <p> By default pinch-to-zoom is enabled.
     *
     * @return True if pinch-to-zoom is enabled.
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    public boolean isPinchToZoomEnabled() {
        Threads.checkMainThread();
        return mPinchToZoomEnabled;
    }

    @SuppressLint("RestrictedApi")
    @MainThread
    public void setPinchToZoomEnabled(boolean enabled) {
        Threads.checkMainThread();
        mPinchToZoomEnabled = enabled;
    }


    @SuppressLint("RestrictedApi")
    @SuppressWarnings("FutureReturnValueIgnored")
    public void onPinchToZoom(float pinchToZoomScale) {
        if (!isCameraAttached()) {
            Logger.w(TAG, CAMERA_NOT_ATTACHED);
            return;
        }
        if (!mPinchToZoomEnabled) {
            Logger.d(TAG, "Pinch to zoom disabled.");
            return;
        }
        Logger.d(TAG, "Pinch to zoom with scale: " + pinchToZoomScale);

        ZoomState zoomState = getZoomState().getValue();
        if (zoomState == null) {
            return;
        }
        float clampedRatio = zoomState.getZoomRatio() * speedUpZoomBy2X(pinchToZoomScale);
        // Clamp the ratio with the zoom range.
        clampedRatio = Math.min(Math.max(clampedRatio, zoomState.getMinZoomRatio()),
                zoomState.getMaxZoomRatio());
        setZoomRatio(clampedRatio);
    }

    private float speedUpZoomBy2X(float scaleFactor) {
        if (scaleFactor > 1f) {
            return 1.0f + (scaleFactor - 1.0f) * 2;
        } else {
            return 1.0f - (1.0f - scaleFactor) * 2;
        }
    }

    /**
     * 点击对焦
     */
    @SuppressLint("RestrictedApi")
    @SuppressWarnings("FutureReturnValueIgnored")
    public void onTapToFocus(MeteringPointFactory meteringPointFactory, float x, float y) {
        if (!isCameraAttached()) {
            Logger.w(TAG, CAMERA_NOT_ATTACHED);
            return;
        }
        if (!mTapToFocusEnabled) {
            Logger.d(TAG, "Tap to focus disabled. ");
            return;
        }
        Logger.d(TAG, "Tap to focus started: start:" + x + ", " + y);
        mTapToFocusState.postValue(TAP_TO_FOCUS_STARTED);
        MeteringPoint afPoint = meteringPointFactory.createPoint(x, y, AF_SIZE);
        MeteringPoint aePoint = meteringPointFactory.createPoint(x, y, AE_SIZE);
        Logger.d(TAG, "Tap to focus started: after:" + afPoint.getX() + ", " + afPoint.getY());

        FocusMeteringAction focusMeteringAction = new FocusMeteringAction
                .Builder(afPoint, FocusMeteringAction.FLAG_AF)
                .addPoint(aePoint, FocusMeteringAction.FLAG_AE)
                .build();
        Futures.addCallback(mCamera.getCameraControl().startFocusAndMetering(focusMeteringAction),
                new FutureCallback<FocusMeteringResult>() {

                    @Override
                    public void onSuccess(@Nullable FocusMeteringResult result) {
                        if (result == null) {
                            return;
                        }
                        Logger.d(TAG, "Tap to focus onSuccess: " + result.isFocusSuccessful());
                        mTapToFocusState.postValue(result.isFocusSuccessful()
                                ? TAP_TO_FOCUS_SUCCESSFUL : TAP_TO_FOCUS_UNSUCCESSFUL);
                    }

                    @Override
                    public void onFailure(Throwable t) {
                        if (t instanceof CameraControl.OperationCanceledException) {
                            Logger.d(TAG, "Tap-to-focus is canceled by new action.");
                            return;
                        }
                        Logger.d(TAG, "Tap to focus failed.", t);
                        mTapToFocusState.postValue(TAP_TO_FOCUS_FAILED);
                    }
                }, CameraXExecutors.directExecutor());
    }

    @SuppressLint("RestrictedApi")
    public ListenableFuture<FocusMeteringResult> focus(MeteringPointFactory meteringPointFactory,
                                                       float x, float y, float ae_size, float af_size, long autoCancelTime /*ms*/) {
        if (!isCameraAttached()) {
            Logger.w(TAG, CAMERA_NOT_ATTACHED);
            return Futures.immediateFailedFuture(new Throwable("camera not attach"));
        }
        Logger.d(TAG, "Silent focus started: start:" + x + ", " + y);
        MeteringPoint afPoint = meteringPointFactory.createPoint(x, y, af_size);
        MeteringPoint aePoint = meteringPointFactory.createPoint(x, y, ae_size);
        Logger.d(TAG, "Silent focus started: after:" + afPoint.getX() + ", " + afPoint.getY());
        FocusMeteringAction focusMeteringAction = new FocusMeteringAction
                .Builder(afPoint, FocusMeteringAction.FLAG_AF)
                .addPoint(aePoint, FocusMeteringAction.FLAG_AE)
                .setAutoCancelDuration(autoCancelTime, TimeUnit.MILLISECONDS)
                .build();
        ListenableFuture<FocusMeteringResult> focusFuture =
                mCamera.getCameraControl().startFocusAndMetering(focusMeteringAction);
        Futures.addCallback(focusFuture, new FutureCallback<FocusMeteringResult>() {

            @Override
            public void onSuccess(@Nullable FocusMeteringResult result) {
                if (result == null) {
                    return;
                }
                Logger.d(TAG, "Silent focus onSuccess: " + result.isFocusSuccessful());
            }

            @Override
            public void onFailure(Throwable t) {
                if (t instanceof CameraControl.OperationCanceledException) {
                    Logger.d(TAG, "Silent-focus is canceled by new action.");
                    return;
                }
                Logger.d(TAG, "Silent focus failed.", t);
            }
        }, CameraXExecutors.directExecutor());

        return focusFuture;
    }

    /**
     * Returns whether tap-to-focus is enabled.
     *
     * <p> By default tap-to-focus is enabled.
     *
     * @return True if tap-to-focus is enabled.
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    public boolean isTapToFocusEnabled() {
        Threads.checkMainThread();
        return mTapToFocusEnabled;
    }


    @SuppressLint("RestrictedApi")
    @MainThread
    public void setTapToFocusEnabled(boolean enabled) {
        Threads.checkMainThread();
        mTapToFocusEnabled = enabled;
    }


    @SuppressLint("RestrictedApi")
    @MainThread
    @NonNull
    public LiveData<Integer> getTapToFocusState() {
        Threads.checkMainThread();
        return mTapToFocusState;
    }

    @SuppressLint("RestrictedApi")
    @NonNull
    @MainThread
    public LiveData<ZoomState> getZoomState() {
        Threads.checkMainThread();
        return mZoomState;
    }

    @SuppressLint("RestrictedApi")
    @Nullable
    @MainThread
    public CameraInfo getCameraInfo() {
        Threads.checkMainThread();
        return mCamera == null ? null : mCamera.getCameraInfo();
    }

    @SuppressLint("RestrictedApi")
    @MainThread
    public LiveData<Observable<CameraInternal.State>> getCameraState() {
        Threads.checkMainThread();
        return mStateLiveData;
    }


    @SuppressLint("RestrictedApi")
    @Nullable
    @MainThread
    public CameraControl getCameraControl() {
        Threads.checkMainThread();
        return mCamera == null ? null : mCamera.getCameraControl();
    }

    @SuppressLint("RestrictedApi")
    @NonNull
    @MainThread
    public ListenableFuture<Void> setZoomRatio(float zoomRatio) {
        Threads.checkMainThread();
        if (!isCameraAttached()) {
            Logger.w(TAG, CAMERA_NOT_ATTACHED);
            return Futures.immediateFuture(null);
        }
        return mCamera.getCameraControl().setZoomRatio(zoomRatio);
    }

    @SuppressLint("RestrictedApi")
    @NonNull
    @MainThread
    public ListenableFuture<Void> setLinearZoom(@FloatRange(from = 0f, to = 1f) float linearZoom) {
        Threads.checkMainThread();
        if (!isCameraAttached()) {
            Logger.w(TAG, CAMERA_NOT_ATTACHED);
            return Futures.immediateFuture(null);
        }
        return mCamera.getCameraControl().setLinearZoom(linearZoom);
    }

    @SuppressLint("RestrictedApi")
    @NonNull
    @MainThread
    public LiveData<Integer> getTorchState() {
        Threads.checkMainThread();
        return mTorchState;
    }

    @SuppressLint("RestrictedApi")
    @NonNull
    @MainThread
    public LiveData<CameraInfo> getCameraInfoLiveData() {
        return mCameraInfoLiveData;
    }

    public MutableLiveData<CameraControl> getCameraControlLiveData() {
        return mCameraControlLiveData;
    }

    @SuppressLint("RestrictedApi")
    @NonNull
    @MainThread
    public ListenableFuture<Void> enableTorch(boolean torchEnabled) {
        Threads.checkMainThread();
        if (!isCameraAttached()) {
            Logger.w(TAG, CAMERA_NOT_ATTACHED);
            return Futures.immediateFuture(null);
        }
        return mCamera.getCameraControl().enableTorch(torchEnabled);
    }

    /**
     * Binds use cases, gets a new {@link Camera} instance and tracks the state of the camera.
     */
    void startCameraAndTrackStates() {
        startCameraAndTrackStates(null);
    }

    @SuppressLint("RestrictedApi")
    void startCameraAndTrackStates(@Nullable Runnable restoreStateRunnable) {
        try {
            mCamera = startCamera();
        } catch (Exception exception) {
            if (restoreStateRunnable != null) {
                restoreStateRunnable.run();
            }
            // Catches the core exception and throw a more readable one.
            String errorMessage =
                    "The selected camera does not support the enabled use cases. Please "
                            + "disable use case and/or select a different camera. e.g. "
                            + "#setVideoCaptureEnabled(false)";
            exception.printStackTrace();
            //throw new IllegalStateException(errorMessage, exception);
        }
        if (!isCameraAttached()) {
            Logger.d(TAG, CAMERA_NOT_ATTACHED);
            return;
        }
        mCameraInfoLiveData.setValue(mCamera.getCameraInfo());
        mCameraControlLiveData.setValue(mCamera.getCameraControl());
        mZoomState.setSource(mCamera.getCameraInfo().getZoomState());
        mTorchState.setSource(mCamera.getCameraInfo().getTorchState());

        for (CameraInternal cameraInternal : mCamera.getCameraInternals()) {
            if (mCamera.getCameraInfo() instanceof CameraInfoInternal && cameraInternal.getCameraInfo() instanceof CameraInfoInternal) {
                String cameraId = ((CameraInfoInternal) mCamera.getCameraInfo()).getCameraId();
                String compressCameraId = ((CameraInfoInternal) cameraInternal.getCameraInfo()).getCameraId();
                if (cameraId.equals(compressCameraId)) {
                    mStateLiveData.setValue(cameraInternal.getCameraState());
                    break;
                }
            }
        }
        Log.i("CameraLifeManager", "init, PreviewSize: " + mPreview.getAttachedSurfaceResolution() + "  " +
                "  CaptureSize:" + mImageCapture.getAttachedSurfaceResolution());

    }

    @SuppressLint({"RestrictedApi", "UnsafeExperimentalUsageError", "UnsafeOptInUsageError"})
    @Nullable
    @RestrictTo(RestrictTo.Scope.LIBRARY_GROUP)
    protected UseCaseGroup createUseCaseGroup() {
        if (!isCameraInitialized()) {
            Logger.d(TAG, CAMERA_NOT_INITIALIZED);
            return null;
        }
        if (!isPreviewViewAttached()) {
            // Preview is required. Return early if preview Surface is not ready.
            Logger.d(TAG, PREVIEW_VIEW_NOT_ATTACHED);
            return null;
        }

        UseCaseGroup.Builder builder = new UseCaseGroup.Builder().addUseCase(mPreview);

        if (isImageCaptureEnabled()) {
            builder.addUseCase(mImageCapture);
        } else {
            mCameraProvider.unbind(mImageCapture);
        }

        if (isImageAnalysisEnabled()) {
            builder.addUseCase(mImageAnalysis);
        } else {
            mCameraProvider.unbind(mImageAnalysis);
        }

        builder.setViewPort(mViewPort);
        return builder.build();
    }

    @SuppressWarnings("WeakerAccess")
    class DisplayRotationListener implements DisplayManager.DisplayListener {

        @Override
        public void onDisplayAdded(int displayId) {
        }

        @Override
        public void onDisplayRemoved(int displayId) {
        }

        @SuppressLint({"WrongConstant", "UnsafeExperimentalUsageError", "UnsafeOptInUsageError"})
        @Override
        public void onDisplayChanged(int displayId) {

        }
    }

    /**
     * Nested class to avoid verification errors for methods introduced in Android 11 (API 30).
     */
    @RequiresApi(30)
    private static class Api30Impl {

        private Api30Impl() {
        }

        @NonNull
        static Context createAttributionContext(@NonNull Context context,
                                                @Nullable String attributeTag) {
            return context.createAttributionContext(attributeTag);
        }

        @Nullable
        static String getAttributionTag(@NonNull Context context) {
            return context.getAttributionTag();
        }
    }

    public static class OutputSize {

        /**
         * A value that represents the aspect ratio is not assigned.
         */
        public static final int UNASSIGNED_ASPECT_RATIO = -1;

        /**
         * Possible value for {@link #getAspectRatio()}
         *
         * @hide
         */
        @RestrictTo(RestrictTo.Scope.LIBRARY)
        @Retention(RetentionPolicy.SOURCE)
        @IntDef(value = {UNASSIGNED_ASPECT_RATIO, AspectRatio.RATIO_4_3, AspectRatio.RATIO_16_9})
        public @interface OutputAspectRatio {
        }

        @OutputAspectRatio
        private final int mAspectRatio;

        @Nullable
        private final Size mResolution;

        /**
         * Creates a {@link OutputSize} that is based on aspect ratio.
         *
         * @see Preview.Builder#setTargetAspectRatio(int)
         * @see ImageAnalysis.Builder#setTargetAspectRatio(int)
         */
        @SuppressLint("RestrictedApi")
        public OutputSize(@AspectRatio.Ratio int aspectRatio) {
            Preconditions.checkArgument(aspectRatio != UNASSIGNED_ASPECT_RATIO);
            mAspectRatio = aspectRatio;
            mResolution = null;
        }

        /**
         * Creates a {@link OutputSize} that is based on resolution.
         *
         * @see Preview.Builder#setTargetResolution(Size)
         * @see ImageAnalysis.Builder#setTargetResolution(Size)
         */
        @SuppressLint("RestrictedApi")
        public OutputSize(@NonNull Size resolution) {
            Preconditions.checkNotNull(resolution);
            mAspectRatio = UNASSIGNED_ASPECT_RATIO;
            mResolution = resolution;
        }

        /**
         * Gets the value of aspect ratio.
         *
         * @return {@link #UNASSIGNED_ASPECT_RATIO} if the size is not based on aspect ratio.
         */
        @OutputAspectRatio
        public int getAspectRatio() {
            return mAspectRatio;
        }

        /**
         * Gets the value of resolution.
         *
         * @return null if the size is not based on resolution.
         */
        @Nullable
        public Size getResolution() {
            return mResolution;
        }
    }

    public void setPreview(@NonNull Preview preview) {
        this.mPreview = preview;
    }

    public void setImageCapture(@NonNull ImageCapture imageCapture) {
        this.mImageCapture = imageCapture;
    }

    @NonNull
    public Preview getPreview() {
        return mPreview;
    }

    @NonNull
    public ImageCapture getImageCapture() {
        return mImageCapture;
    }

    @SuppressLint("RestrictedApi")
    public int getCaptureTargetRotation() {
        CameraInternal attachedCamera = mImageCapture.getCamera();
        if (attachedCamera == null) {
            return 90;
        }

        return attachedCamera.getCameraInfoInternal().getSensorRotationDegrees(
                mImageCapture.getTargetRotation());
    }
}
