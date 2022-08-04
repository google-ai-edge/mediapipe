package com.ola.olamera.camera.session;


import android.os.Handler;
import android.os.Looper;

import androidx.annotation.NonNull;

import com.ola.olamera.camera.camera.CameraErrorListenerHandlerWrapper;
import com.ola.olamera.camera.camera.ICameraErrorListener;
import com.ola.olamera.camera.session.config.CameraSelectConfig;


public class SessionConfig {


    private final PreviewConfig mPreviewConfig;

    private ImageCapture mCaptureConfig;

    private ICameraErrorListener mCameraErrorListener;

    private CameraSelectConfig mSelectConfig;


    public SessionConfig(@NonNull PreviewConfig previewConfig) {
        mPreviewConfig = previewConfig;
    }

    public void setCaptureConfig(ImageCapture captureConfig) {
        mCaptureConfig = captureConfig;
    }

    public void setSelectConfig(CameraSelectConfig selectConfig) {
        mSelectConfig = selectConfig;
    }

    public @NonNull
    PreviewConfig getPreviewConfig() {
        return mPreviewConfig;
    }

    public ImageCapture getImageCapture() {
        return mCaptureConfig;
    }

    public CameraSelectConfig getSelectConfig() {
        return mSelectConfig;
    }

    public void setCameraErrorListener(@NonNull ICameraErrorListener cameraErrorListener, Handler handler) {
        if (handler == null) {
            mCameraErrorListener = cameraErrorListener;
        } else {
            mCameraErrorListener = new CameraErrorListenerHandlerWrapper(new Handler(Looper.getMainLooper()), cameraErrorListener);
        }
    }

    public ICameraErrorListener getCameraErrorListener() {
        return mCameraErrorListener;
    }
}
