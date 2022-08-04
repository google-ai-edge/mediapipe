package com.ola.olamera.camerax.controller;

import android.util.Size;
import android.view.MotionEvent;

import androidx.annotation.UiThread;
import androidx.camera.core.FocusMeteringResult;
import androidx.camera.core.SurfaceRequest;

import com.google.common.util.concurrent.ListenableFuture;
import com.ola.olamera.camerax.CameraController;

/**
 * @author : liujian
 * @date : 2021/7/29
 */
public interface ICameraViewTouchManager {

    @UiThread
    void setCameraController(CameraController cameraController);

    boolean onTouchEvent(MotionEvent event);

    void performClick();

    void onAttachedToWindow();

    void onDetachedFromWindow();

    void onSizeChanged();

    void setTransformationInfo(SurfaceRequest.TransformationInfo transformationInfo, Size resolution, boolean isFrontCamera);

    void setOnGestureDetectorListener(OnGestureDetectorListener listener);

    ListenableFuture<FocusMeteringResult> autoFocus(float x, float y, float size, long autoCancelTimes /*ms*/);

}
