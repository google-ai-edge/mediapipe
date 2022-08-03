package com.quark.quamera.camerax;

import android.util.Size;

import com.quark.quamera.camera.session.SingleCaptureConfig;
import com.quark.quamera.render.view.BasePreviewView;

import androidx.annotation.NonNull;

/**
 * @author : yushan.lj@alibaba-inc.com
 * @date : 2021/10/17
 */
public interface ICameraManager<T extends BasePreviewView> {

    void setCameraPreview(@NonNull T cameraPreview);

    void onWindowCreate();

    void onWindowActive();

    void onWindowInactive();

    void onWindowDestroy();

    void startCamera(Size previewSize, boolean useWideCamera, boolean needBindLifecycle);

    void startCamera(Size previewSize, boolean useWideCamera, boolean needBindLifecycle, boolean isLimitCaptureSize, Size maxCaptureSize);

    void switchCamera(com.quark.quamera.camera.session.CameraSelector.CameraLenFacing cameraLenFacing, Size previewSize);

    void takePictureOriginalData(SingleCaptureConfig singleCaptureConfig, com.quark.quamera.camera.session.ImageCapture.OnImageCapturedCallback capturedCallback);

    void enableFlash(boolean enable);


}
