package com.ola.olamera.render.view;


import android.annotation.SuppressLint;
import android.content.Context;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import android.util.Size;
import android.view.MotionEvent;

import com.google.common.util.concurrent.ListenableFuture;
import com.ola.olamera.camera.preview.IPreviewSurfaceProvider;
import com.ola.olamera.camera.preview.SurfaceTextureWrapper;
import com.ola.olamera.camerax.CameraController;
import com.ola.olamera.camerax.controller.CameraViewTouchManager;
import com.ola.olamera.camerax.controller.ICameraViewTouchManager;
import com.ola.olamera.camerax.controller.OnGestureDetectorListener;
import com.ola.olamera.render.DefaultCameraRender;
import com.ola.olamera.util.CameraLogger;

import androidx.annotation.AnyThread;
import androidx.annotation.GuardedBy;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.camera.camera2.internal.Camera2CameraCaptureResult;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.FocusMeteringResult;
import androidx.camera.core.Logger;
import androidx.camera.core.Preview;
import androidx.camera.core.SurfaceRequest;
import androidx.camera.core.impl.CameraInternal;
import androidx.core.content.ContextCompat;

/**
 * CameraX的预览View
 */
public class CameraXPreviewView extends BasePreviewView {
    private static final String TAG = "CameraPreviewView";

    private ICameraViewTouchManager mManager;
    // @GuardedBy("mSurfaceLock")
    private DefaultCameraRender mDefaultCameraRender;


    public CameraXPreviewView(@NonNull Context context, @Nullable AttributeSet attrs) {
        this(context, false);
    }

    public CameraXPreviewView(Context context, boolean ZOderOverlay) {
        super(context, ZOderOverlay);
        mManager = new CameraViewTouchManager(this);
    }

    @Override
    public void setOnGestureDetectorListener(OnGestureDetectorListener listener) {
        mManager.setOnGestureDetectorListener(listener);
    }

    private final OnLayoutChangeListener mOnLayoutChangeListener =
            (v, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom) -> {
                boolean isSizeChanged =
                        right - left != oldRight - oldLeft || bottom - top != oldBottom - oldTop;
                if (isSizeChanged) {
                    mManager.onSizeChanged();
                }
            };

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        addOnLayoutChangeListener(mOnLayoutChangeListener);
        mManager.onAttachedToWindow();
    }

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        removeOnLayoutChangeListener(mOnLayoutChangeListener);
        mManager.onDetachedFromWindow();
    }

    /**
     * 用于相机缩放，轻点对焦的逻辑处理
     */
    public void setCameraController(CameraController cameraController) {
        mManager.setCameraController(cameraController);
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        return mManager.onTouchEvent(event) || super.onTouchEvent(event);
    }

    @Override
    public boolean performClick() {
        mManager.performClick();
        return super.performClick();
    }

    @Override
    public IPreviewSurfaceProvider getSurfaceProvider() {
        return mPreviewSurfaceProvider;
    }

    @Override
    public ListenableFuture<FocusMeteringResult> autoFocus(float x, float y, float size, long autoCancelTime) {
        return mManager.autoFocus(x, y, size, autoCancelTime);
    }

    final IPreviewSurfaceProvider mPreviewSurfaceProvider = new IPreviewSurfaceProvider() {
        @Override
        public Preview.SurfaceProvider providerSurface() {
            return mSurfaceProvider;
        }
    };

    final Preview.SurfaceProvider mSurfaceProvider = new Preview.SurfaceProvider() {

        @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
        @SuppressLint({"RestrictedApi", "UnsafeOptInUsageError", "UnsafeExperimentalUsageError"})
        @Override
        @AnyThread
        public void onSurfaceRequested(@NonNull SurfaceRequest request) {
            post(() -> {
                releaseSurface();

                Log.i(TAG, "onSurfaceRequested  provideSurface");
                CameraInternal camera = request.getCamera();
                request.setTransformationInfoListener(
                        ContextCompat.getMainExecutor(getContext()),
                        transformationInfo -> {
                            Logger.d(TAG, "Preview transformation info updated. " + transformationInfo);
                            camera.getCameraInfoInternal();

                            Integer lensFacing = camera.getCameraInfoInternal().getLensFacing();
                            boolean isFrontCamera = lensFacing != null &&
                                    lensFacing == CameraSelector.LENS_FACING_FRONT;
                            mManager.setTransformationInfo(transformationInfo,
                                    request.getResolution(), isFrontCamera);
                        });

                Size size = request.getResolution();
                CameraLogger.i("CameraLifeManager", "onSurfaceRequested: previewWidth:" + size.getWidth() + "  height:" + size.getHeight());
                //判断是否相同surface，不同重新创建 相同不处理
                SurfaceTextureWrapper surfaceTextureWrapper = new SurfaceTextureWrapper(size.getWidth(), size.getHeight());
                mDefaultCameraRender = new DefaultCameraRender(surfaceTextureWrapper, null, null);
                getRender().setCameraRender(mDefaultCameraRender);
                request.provideSurface(surfaceTextureWrapper.getSurface(),
                        getGLExecutor(), result1 -> {
                            // 判断是否相同surface
                        });

            });
        }
    };

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void releaseSurface() {
        if (mDefaultCameraRender != null) {
            mDefaultCameraRender.destroySurface();
            mDefaultCameraRender = null;
        }
    }

    public void cacheCaptureResult(Camera2CameraCaptureResult captureResult) {
        if (mDefaultCameraRender != null) {
            mDefaultCameraRender.cacheCameraXCaptureResult(captureResult);
        }
    }
}