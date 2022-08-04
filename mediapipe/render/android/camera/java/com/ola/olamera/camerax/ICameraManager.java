package com.ola.olamera.camerax;

import android.util.Size;

import com.ola.olamera.camera.session.SingleCaptureConfig;
import com.ola.olamera.render.view.BasePreviewView;

import androidx.annotation.NonNull;

/**
 * @author : yushan.lj@
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

    void switchCamera(com.ola.olamera.camera.session.CameraSelector.CameraLenFacing cameraLenFacing, Size previewSize);

    void takePictureOriginalData(SingleCaptureConfig singleCaptureConfig, com.ola.olamera.camera.session.ImageCapture.OnImageCapturedCallback capturedCallback);

    void enableFlash(boolean enable);


}
