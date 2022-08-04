package com.ola.olamera.camerax.controller;

import android.annotation.SuppressLint;
import android.graphics.PointF;
import android.os.Build;
import android.util.Log;
import android.util.Rational;
import android.util.Size;
import android.view.Display;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.ViewConfiguration;

import com.google.common.util.concurrent.ListenableFuture;
import com.ola.olamera.camerax.CameraController;
import com.ola.olamera.render.view.CameraXPreviewView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.annotation.UiThread;
import androidx.camera.core.FocusMeteringResult;
import androidx.camera.core.SurfaceRequest;
import androidx.camera.core.ViewPort;
import androidx.camera.core.ZoomState;
import androidx.camera.core.impl.ImageOutputConfig;
import androidx.camera.core.impl.utils.futures.Futures;

/**
 * @author : liujian
 * @date : 2021/7/28
 */
public class CameraViewTouchManager implements ICameraViewTouchManager {
    private static final String TAG = "CameraViewTouchListener";

    private final CameraXPreviewView mTouchView;
    private final ScaleGestureDetector mScaleGestureDetector;

    @NonNull
    final PreviewTransformation mPreviewTransform = new PreviewTransformation();

    @NonNull
    PreviewViewMeteringPointFactory mPreviewViewMeteringPointFactory =
            new PreviewViewMeteringPointFactory(mPreviewTransform);

    @Nullable
    private MotionEvent mTouchUpEvent;

    private OnGestureDetectorListener mGestureListener;
    private CameraController mCameraController;

    class PinchToZoomOnScaleGestureListener extends
            ScaleGestureDetector.SimpleOnScaleGestureListener {
        @Override
        public boolean onScale(ScaleGestureDetector detector) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
                return false;
            }
            if (mCameraController != null) {
                // 手势缩放
                mCameraController.onPinchToZoom(detector.getScaleFactor());

                ZoomState value = mCameraController.getZoomState().getValue();
                if (mGestureListener != null && value != null) {
                    mGestureListener.onPinchToZoom(value.getZoomRatio(), value.getMaxZoomRatio(), value.getMinZoomRatio());
                }
            }
            return true;
        }
    }

    public CameraViewTouchManager(@NonNull CameraXPreviewView touchView) {
        this.mTouchView = touchView;

        mScaleGestureDetector = new ScaleGestureDetector(
                touchView.getContext(), new PinchToZoomOnScaleGestureListener());
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @UiThread
    @Override
    public void setCameraController(@Nullable CameraController cameraController) {
        if (mCameraController != null && mCameraController != cameraController) {
            mCameraController.clearPreviewSurface();
        }
        mCameraController = cameraController;
        attachToControllerIfReady();
    }

    @Override
    public void setOnGestureDetectorListener(OnGestureDetectorListener listener) {
        this.mGestureListener = listener;
    }

    @SuppressLint("RestrictedApi")

    @Override
    public ListenableFuture<FocusMeteringResult> autoFocus(float x, float y, float size, long autoCancelTimes) {
        if (mCameraController == null) {
            return Futures.immediateFailedFuture(new Throwable("camera controller not init"));
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            return mCameraController.focus(mPreviewViewMeteringPointFactory, x, y, size, size, autoCancelTimes);
        } else {
            return Futures.immediateFailedFuture(new Throwable("api is lower than  Build.VERSION_CODES.LOLLIPOP"));
        }
    }

    @UiThread
    @Nullable
    public ViewPort getViewPort() {
        if (mTouchView.getDisplay() == null) {
            return null;
        }
        return getViewPort(mTouchView.getDisplay().getRotation());
    }


    @UiThread
    @SuppressLint({"WrongConstant", "UnsafeExperimentalUsageError", "UnsafeOptInUsageError"})
    @Nullable
    public ViewPort getViewPort(@ImageOutputConfig.RotationValue int targetRotation) {
        if (mTouchView.getWidth() == 0 || mTouchView.getHeight() == 0) {
            return null;
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
            return null;
        }
        return new ViewPort.Builder(new Rational(mTouchView.getWidth(), mTouchView.getHeight()), targetRotation)
                .setScaleType(getViewPortScaleType())
                .setLayoutDirection(mTouchView.getLayoutDirection())
                .build();
    }


    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (mCameraController == null) {
            return false;
        }
        boolean isSingleTouch = event.getPointerCount() == 1;
        boolean isUpEvent = event.getAction() == MotionEvent.ACTION_UP;
        boolean notALongPress = event.getEventTime() - event.getDownTime()
                < ViewConfiguration.getLongPressTimeout();
        if (isSingleTouch && isUpEvent && notALongPress) {
            mTouchUpEvent = event;
            performClick();

            return true;
        }
        return mScaleGestureDetector.onTouchEvent(event);
    }

    @Override
    public void performClick() {
        if (mCameraController != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            float x = mTouchUpEvent != null ? mTouchUpEvent.getX() : mTouchView.getWidth() / 2f;
            float y = mTouchUpEvent != null ? mTouchUpEvent.getY() : mTouchView.getHeight() / 2f;

            float[] marginPercentage = mTouchView.getRender().getMarginPercentage();

            float viewHeight = mTouchView.getHeight();
            float topY = viewHeight * marginPercentage[1];
            float bottomY = viewHeight * (1 - marginPercentage[3]);
            if (viewHeight == 0 || (y >= topY && y <= bottomY)) {
                // 点击对焦
                mCameraController.onTapToFocus(mPreviewViewMeteringPointFactory, x, y);

                if (mGestureListener != null) {
                    mGestureListener.onClickFocused(x, y);
                }
            }
        }
        mTouchUpEvent = null;
    }

    @Override
    public void onAttachedToWindow() {
        attachToControllerIfReady();
    }

    @Override
    public void onDetachedFromWindow() {
        if (mCameraController != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            mCameraController.clearPreviewSurface();
        }
    }

    /**
     * 视图的Size变换后，重新给对焦类重新计算视图宽高
     */
    @Override
    public void onSizeChanged() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
            return;
        }
        mPreviewViewMeteringPointFactory.recalculate(new Size(mTouchView.getWidth(), mTouchView.getHeight()),
                mTouchView.getLayoutDirection());
        attachToControllerIfReady();
    }

    @Override
    public void setTransformationInfo(SurfaceRequest.TransformationInfo transformationInfo, Size resolution, boolean isFrontCamera) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
            return;
        }
        mPreviewTransform.setTransformationInfo(transformationInfo,
                resolution, isFrontCamera);
        mPreviewViewMeteringPointFactory.recalculate(new Size(mTouchView.getWidth(), mTouchView.getHeight()),
                mTouchView.getLayoutDirection());
    }


    private void attachToControllerIfReady() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
            return;
        }
        Display display = mTouchView.getDisplay();
        ViewPort viewPort = getViewPort();
        if (mCameraController != null && viewPort != null && mTouchView.isAttachedToWindow()
                && display != null) {
            try {
                mCameraController.attachPreviewSurface(mTouchView.getSurfaceProvider().providerSurface(), viewPort, display);
            } catch (IllegalStateException ex) {
                Log.e(TAG, ex.getMessage(), ex);
            }
        }
    }

    private int getViewPortScaleType() {
        return ViewPort.FILL_CENTER;
    }

}
