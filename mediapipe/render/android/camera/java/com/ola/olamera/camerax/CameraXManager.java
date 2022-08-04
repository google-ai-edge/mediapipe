package com.ola.olamera.camerax;
/*
 *
 *  Creation    :  20-11-25
 *  Author      : jiaming.wjm@
 */

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Rect;
import android.graphics.RectF;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CaptureRequest;
import android.os.Build;
import android.os.Process;
import android.util.LayoutDirection;
import android.util.Log;
import android.util.Size;
import android.util.SizeF;

import com.ola.olamera.camera.preview.ViewPort;
import com.ola.olamera.camera.preview.ViewPorts;
import com.ola.olamera.camera.session.SingleCaptureConfig;
import com.ola.olamera.camerax.utils.FocalLengthInfo;
import com.ola.olamera.camerax.utils.SingleThreadHandlerExecutor;
import com.ola.olamera.render.view.CameraXPreviewView;
import com.ola.olamera.util.CameraLogger;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import androidx.annotation.FloatRange;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.annotation.UiThread;
import androidx.camera.camera2.impl.Camera2ImplConfig;
import androidx.camera.camera2.internal.Camera2CameraCaptureResult;
import androidx.camera.camera2.internal.Camera2CameraControlImpl;
import androidx.camera.camera2.internal.Camera2CameraInfoImpl;
import androidx.camera.core.CameraControl;
import androidx.camera.core.CameraInfo;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageInfo;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.ZoomState;
import androidx.camera.core.impl.CameraCaptureCallback;
import androidx.camera.core.impl.CameraCaptureResult;
import androidx.camera.core.impl.CameraInfoInternal;
import androidx.camera.core.impl.CameraInternal;
import androidx.camera.core.impl.Observable;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.Lifecycle;
import androidx.lifecycle.LifecycleOwner;
import androidx.lifecycle.LifecycleRegistry;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

public class CameraXManager implements LifecycleOwner, ICameraManager<CameraXPreviewView> {

    private LifecycleCameraController mCameraController;
    private CameraXPreviewView mPreviewView;

    private final ExecutorService mImageCaptureExecutorService;

    private final LifecycleRegistry mLifecycleRegistry = new LifecycleRegistry(this);

    static final CameraSelector BACK_SELECTOR =
            new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
    static final CameraSelector FRONT_SELECTOR =
            new CameraSelector.Builder().requireLensFacing(
                    CameraSelector.LENS_FACING_FRONT).build();

    private CameraSelector mCurrentCameraSelector = BACK_SELECTOR;

    private final Context mContext;
    private boolean useWideCamera;
    private final SingleThreadHandlerExecutor mHandlerExecutor;

    private final MutableLiveData<Integer> mCameraStateLiveData = new MutableLiveData<>();
    private CameraInfoInternal mCameraInfoInternal;

    private Observable<CameraInternal.State> mLastObservable;

    private boolean isLimitCaptureSize = true;
    private Size maxCaptureSize;

    public CameraXManager(@NonNull Context context) {
        this.mContext = context;
        mImageCaptureExecutorService = Executors.newSingleThreadExecutor();
        mHandlerExecutor = new SingleThreadHandlerExecutor("camera_capture_result",
                Process.THREAD_PRIORITY_BACKGROUND);
    }

    public void setCameraPreview(@NonNull CameraXPreviewView cameraPreview) {
        mPreviewView = cameraPreview;
    }

    @Override
    public void startCamera(Size previewSize, boolean useWideCamera, boolean needBindLifecycle) {
        startCamera(previewSize, useWideCamera, needBindLifecycle, true, null);
    }

    @SuppressLint("RestrictedApi")
    public void startCamera(Size previewSize, boolean useWideCamera, boolean needBindLifecycle, boolean isLimitCaptureSize, Size maxCaptureSize) {
        this.useWideCamera = useWideCamera;
        this.isLimitCaptureSize = isLimitCaptureSize;
        this.maxCaptureSize = maxCaptureSize;

        if (mLastObservable != null) {
            mLastObservable.removeObserver(observer);
        }
        if (mCameraController != null) {
            mCameraController.unbind();
        }

        mCameraController = new LifecycleCameraController(mContext, previewSize, isLimitCaptureSize, maxCaptureSize, new OnCameraProviderListener() {
            @Override
            public void onProcessCameraProvider(ProcessCameraProvider provider) {
                mCameraController.initCameraSelector(mCurrentCameraSelector);
            }
        });
        mCameraController.getCameraInfoLiveData().observe(this, this::onCameraInfoObserved);
        if (needBindLifecycle) {
            mCameraController.bindToLifecycle(this);
        }
        /*
         * 内部会将CameraPreviewView的mSurfaceProvider对象传给Preview的setSurfaceProvider方法
         * 然后再调用SurfaceRequest的providerSurface方法将Surface传入
         */
        mPreviewView.setCameraController(mCameraController);

        mCameraController.getCameraState().observeForever(stateObservable -> {
            this.mLastObservable = stateObservable;
            stateObservable.addObserver(ContextCompat.getMainExecutor(mContext), observer);
        });
    }

    private CameraInternal.State mPreState;

    private final Observable.Observer<CameraInternal.State> observer = new Observable.Observer<CameraInternal.State>() {
        @Override
        public void onNewData(@Nullable CameraInternal.State value) {
            if (value == CameraInternal.State.OPEN) {
                mCameraStateLiveData.setValue(1);
            } else if (value == CameraInternal.State.CLOSED) {
                // 正在关闭相机，下一帧的时候释放Surface
                if (CameraInternal.State.CLOSING == mPreState) {
                    Log.i("CameraXManager", "releaseSurface: ");
                    mPreviewView.post(() -> mPreviewView.releaseSurface());
                }
            }
            mPreState = value;
        }

        @Override
        public void onError(@NonNull Throwable t) {

        }
    };


    @UiThread
    public void takePicture(ImageCapture.OutputFileOptions outputFileOptions,
                            @NonNull ImageCapture.OnImageSavedCallback imageSavedCallback) {
        if (mCameraController == null) {
            return;
        }
        // 添加viewPortCropRect对象，拍照裁剪的Rect
        mCameraController.takePicture(outputFileOptions, mImageCaptureExecutorService, imageSavedCallback, getViewPortRect());
    }

    /**
     * 拍照返回原始数据，没有对数据进行裁剪和旋转操作
     */
    @UiThread
    @SuppressLint("RestrictedApi")
    public void takePictureOriginalData(SingleCaptureConfig singleCaptureConfig, com.ola.olamera.camera.session.ImageCapture.OnImageCapturedCallback capturedCallback) {
        if (mCameraController == null) {
            return;
        }
        Rect viewPortRect;
        if (singleCaptureConfig.getCameraShowRect() != null) {
            RectF rectF = singleCaptureConfig.getCameraShowRect();
            viewPortRect = new Rect((int) rectF.left, (int) rectF.top, (int) rectF.right, (int) rectF.bottom);
        } else {
            viewPortRect = getViewPortRect();
        }
        if (viewPortRect != null) {
            mCameraController.getImageCapture().setViewPortCropRect(viewPortRect);
        }
        mCameraController.takePicture(mImageCaptureExecutorService, new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                super.onCaptureSuccess(image);
                try {
                    ImageProxy.PlaneProxy[] planes = image.getPlanes();
                    if (planes.length > 0 && capturedCallback != null) {
                        ByteBuffer buffer = planes[0].getBuffer();
                        // 原始数据
                        byte[] bytes = new byte[buffer.capacity()];
                        buffer.get(bytes);

                        Size size = new Size(Math.max(image.getHeight(), image.getWidth()), Math.min(image.getWidth(), image.getHeight()));
                        ImageInfo imageInfo = image.getImageInfo();
                        // 原始数据回调给外层处理
                        capturedCallback.onCaptureSuccess(bytes, size, image.getCropRect(), ExifRotationHelper.isHuaweiNova3i() ?
                                mCameraController.getCaptureTargetRotation() : imageInfo.getRotationDegrees());
                    }
                } finally {
                    image.close();
                }
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                super.onError(exception);
                if (capturedCallback != null) {
                    capturedCallback.onError(exception);
                }
            }
        });
    }

    private com.ola.olamera.camera.session.CameraSelector.CameraLenFacing mPreCameraLenFacing =
            com.ola.olamera.camera.session.CameraSelector.CameraLenFacing.LEN_FACING_BACK;

    /**
     * 转换摄像头
     *
     * @param cameraLenFacing CameraLenFacing
     */
    @UiThread
    @SuppressLint("RestrictedApi")
    public void switchCamera(com.ola.olamera.camera.session.CameraSelector.CameraLenFacing cameraLenFacing, Size previewSize) {
        if (cameraLenFacing == null) {
            if (mCurrentCameraSelector == FRONT_SELECTOR) {
                mCurrentCameraSelector = BACK_SELECTOR;
            } else {
                mCurrentCameraSelector = FRONT_SELECTOR;
            }
        } else {
            if (mPreCameraLenFacing != null && mPreCameraLenFacing == cameraLenFacing) {
                return;
            }
            if (cameraLenFacing == com.ola.olamera.camera.session.CameraSelector.CameraLenFacing.LEN_FACING_FONT) {
                mCurrentCameraSelector = FRONT_SELECTOR;
            } else {
                mCurrentCameraSelector = BACK_SELECTOR;
            }
            mPreCameraLenFacing = cameraLenFacing;
        }
        startCamera(previewSize, useWideCamera, true, isLimitCaptureSize, maxCaptureSize);
    }

    @SuppressLint("RestrictedApi")
    public void enableFlash(boolean enable) {
        if (mCameraController == null || !mCameraController.isCameraAttached()) {
            return;
        }
        CameraInfo cameraInfo = getCameraInfo();
        CameraControl cameraControl = getCameraControl();
        if (cameraInfo == null || cameraControl == null) {
            return;
        }
        cameraControl.enableTorch(enable);
    }

    public void setScaleType(@ViewPort.ScaleType int scaleType) {
        mPreviewView.setScaleType(scaleType);
    }

    public void onWindowCreate() {
        mLifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_CREATE);
    }

    public void onWindowActive() {
        mLifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_START);
        mLifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_RESUME);

        if (mCameraController != null) {
            mCameraController.bindToLifecycle(this);
        }
    }

    public void onWindowInactive() {
        mLifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_STOP);
        mLifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_PAUSE);

        if (mCameraController != null) {
            mCameraController.unbind();
        }
    }

    public void onWindowResume() {
        mLifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_RESUME);
    }

    public void onWindowPause() {
        mLifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_PAUSE);
    }

    @SuppressLint("RestrictedApi")
    public void onWindowDestroy() {
        onWindowInactive();
        mLifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_DESTROY);
        if (mImageCaptureExecutorService != null) {
            mImageCaptureExecutorService.shutdown();
        }
        if (mCameraInfoInternal != null) {
            mCameraInfoInternal.removeSessionCaptureCallback(callback);
        }
        if (mHandlerExecutor != null) {
            mHandlerExecutor.shutdown();
        }
        if (mCameraController != null) {
            mCameraController.unbind();
        }
        if (mLastObservable != null) {
            mLastObservable.removeObserver(observer);
        }
    }

    @Nullable
    public CameraControl getCameraControl() {
        return mCameraController != null ? mCameraController.getCameraControl() : null;
    }

    @Nullable
    public CameraInfo getCameraInfo() {
        return mCameraController != null ? mCameraController.getCameraInfo() : null;
    }

    @Nullable
    public LiveData<CameraInfo> getCameraInfoLiveData() {
        return mCameraController != null ? mCameraController.getCameraInfoLiveData() : null;
    }

    public LiveData<Integer> getCameraState() {
        return mCameraStateLiveData;
    }

    public ImageCapture getImageCapture() {
        return mCameraController != null ? mCameraController.getImageCapture() : null;
    }


    public Preview getPreview() {
        return mCameraController != null ? mCameraController.getPreview() : null;
    }

    public void setLinearZoom(@FloatRange(from = 0f, to = 1f) float linearZoom) {
        if (mCameraController == null) {
            return;
        }
        mCameraController.setLinearZoom(linearZoom);
    }

    public void setZoomRatio(float zoomRatio) {
        if (mCameraController == null) {
            return;
        }
        mCameraController.setZoomRatio(zoomRatio);
    }

    @Nullable
    public LiveData<ZoomState> getZoomState() {
        if (mCameraController == null) {
            return null;
        }
        return mCameraController.getZoomState();
    }

    @Nullable
    public LiveData<Integer> getTapToFocusState() {
        if (mCameraController == null) {
            return null;
        }
        return mCameraController.getTapToFocusState();
    }

    @NonNull
    @Override
    public Lifecycle getLifecycle() {
        return mLifecycleRegistry;
    }

    @SuppressLint("RestrictedApi")
    private Rect getViewPortRect() {
        CameraInfo cameraInfo = getCameraInfo();
        Size size = getImageCapture().getAttachedSurfaceResolution();

        if (cameraInfo instanceof Camera2CameraInfoImpl && size != null) {
            Camera2CameraInfoImpl infoImpl = (Camera2CameraInfoImpl) cameraInfo;
            CameraCharacteristics characteristics = infoImpl.getCameraCharacteristicsCompat().toCameraCharacteristics();

            boolean isFrontCamera =
                    characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT;
            Rect sensorRect = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
            Integer sensorOrientation =
                    characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            int rotation = sensorOrientation != null ? sensorOrientation : 0;
            return ViewPorts.calculateViewPortRect(
                    sensorRect,
                    isFrontCamera,
                    mPreviewView.getAspectRatio(),
                    rotation,
                    ViewPort.FILL_CENTER,
                    LayoutDirection.LTR,
                    size
            );
        }
        return null;
    }

    @SuppressLint("RestrictedApi")
    private void onCameraInfoObserved(CameraInfo cameraInfo) {
        // 使用广角镜头的逻辑
        onUseWipeCameraInfo(cameraInfo);

        // 添加每一帧的回调
        addSessionCaptureCallback(cameraInfo);

        // 对焦状态
        onCameraCaptureResulted(cameraInfo);

        //人脸拍摄
    }

    private final CameraCaptureCallback callback = new CameraCaptureCallback() {
        @SuppressLint("RestrictedApi")
        @Override
        public void onCaptureCompleted(@NonNull CameraCaptureResult cameraCaptureResult) {
            super.onCaptureCompleted(cameraCaptureResult);

            // 获取CaptureResult对象。用于每一帧主动释放内存，解决内存峰值高的问题
            if (mPreviewView != null && cameraCaptureResult instanceof Camera2CameraCaptureResult) {
                Camera2CameraCaptureResult camera2CameraCaptureResult = (Camera2CameraCaptureResult) cameraCaptureResult;

                mPreviewView.cacheCaptureResult(camera2CameraCaptureResult);
            }
        }
    };

    @SuppressLint("RestrictedApi")
    private void addSessionCaptureCallback(CameraInfo cameraInfo) {
        if (cameraInfo instanceof CameraInfoInternal) {

            mCameraInfoInternal = ((CameraInfoInternal) cameraInfo);
            mCameraInfoInternal.addSessionCaptureCallback(mHandlerExecutor, callback);
        }
    }

    private final List<FocalLengthInfo> mFocalLengths = new ArrayList<>();
    // 最大的水平广角 超过认为是超广角 可能会导致变形
    private static final int MAX_ANGLE = 90;

    private void onUseWipeCameraInfo(CameraInfo cameraInfo) {
        mFocalLengths.clear();
        if (cameraInfo instanceof Camera2CameraInfoImpl) {
            @SuppressLint("RestrictedApi") CameraCharacteristics cc = ((Camera2CameraInfoImpl) cameraInfo)
                    .getCameraCharacteristicsCompat().toCameraCharacteristics();
            SizeF size = cc.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
            float[] focalLens = cc.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);

            double width = size.getWidth();
            double height = size.getHeight();

            boolean firstOne = true;
            for (float focalLength : focalLens) {
                FocalLengthInfo lengthInfo = new FocalLengthInfo();
                lengthInfo.horizontalAngle = (2 * Math.atan(width / (focalLength * 2)));
                lengthInfo.verticalAngle = (2 * Math.atan(height / (focalLength * 2)));
                lengthInfo.focalLength = focalLength;
                //一般情况下，第一个焦距信息为默认焦距
                lengthInfo.isDefaultFocal = firstOne;
                firstOne = false;
                mFocalLengths.add(lengthInfo);
            }
        }
        Collections.sort(mFocalLengths);

        StringBuilder sb = new StringBuilder("\n");
        for (FocalLengthInfo minFL : mFocalLengths) {
            sb.append(String.format(Locale.CHINA, "camera min_fl: %.2f ; max_h_angle : %.2f ",
                    (minFL != null ? minFL.focalLength : -1),
                    (minFL != null ? minFL.horizontalAngle * 180 / Math.PI : -1)))
                    .append("\n").append(" isDefault:").append((minFL != null && minFL.isDefaultFocal));
        }
        CameraLogger.i("CameraLifeManager", "print match camera list\n %s", sb);

        mCameraController.getCameraControlLiveData().observe(this, this::useWideCameraInfo);
    }

    /**
     * 优先使用逻辑广角镜头
     */
    @SuppressLint({"RestrictedApi", "UnsafeOptInUsageError", "UnsafeExperimentalUsageError"})
    private void useWideCameraInfo(CameraControl cameraControl) {
        FocalLengthInfo bestFocalInfo;
        if (useWideCamera && (bestFocalInfo = getBestFocalInfo()) != null) {
            float focalLength = bestFocalInfo.focalLength;
            if (cameraControl instanceof Camera2CameraControlImpl) {
                Camera2ImplConfig captureRequestOption = new Camera2ImplConfig.Builder()
                        .setCaptureRequestOption(CaptureRequest.LENS_FOCAL_LENGTH, focalLength).build();

                Camera2CameraControlImpl controlImpl = (Camera2CameraControlImpl) cameraControl;
                controlImpl.getCamera2CameraControl().
                        addCaptureRequestOptions(captureRequestOption);
            }
        }
    }

    /**
     * 返回最佳的焦距信息
     */
    private FocalLengthInfo getBestFocalInfo() {
        FocalLengthInfo defaultInfo = null;
        for (FocalLengthInfo info : mFocalLengths) {
            if (info.isDefaultFocal) {
                defaultInfo = info;
            }
            double angle = info.horizontalAngle * 180 / Math.PI;
            if (angle <= MAX_ANGLE) {
                return info;
            }
        }
        // 如果所有焦距对应角度都大于90 选择默认焦距
        return defaultInfo;
    }

    private OnCaptureResultListener mCaptureResultListener;

    public void setOnCaptureResultListener(OnCaptureResultListener listener) {
        this.mCaptureResultListener = listener;
    }

    /**
     * 获取每一帧的对焦状态
     */
    @SuppressLint("RestrictedApi")
    private void onCameraCaptureResulted(CameraInfo cameraInfo) {
        if (cameraInfo instanceof CameraInfoInternal && mCaptureResultListener != null) {
            CameraInfoInternal infoInternal = (CameraInfoInternal) cameraInfo;
            infoInternal.addSessionCaptureCallback(mImageCaptureExecutorService, new CameraCaptureCallback() {
                @Override
                public void onCaptureCompleted(@NonNull CameraCaptureResult cameraCaptureResult) {
                    if (mCaptureResultListener != null) {
                        mCaptureResultListener.onCaptureResulted(cameraCaptureResult);
                    }
                }
            });
        }
    }


}
