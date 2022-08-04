package com.ola.olamera.camera.session;
/*
 *
 *  Creation    :  20-12-19
 *  Author      : jiaming.wjm@
 */

import android.os.Handler;

import androidx.annotation.NonNull;


public class CameraCaptureCallbackHandlerWrapper extends CameraCaptureCallback {
    private Handler mHandler;
    private CameraCaptureCallback mCallback;

    public CameraCaptureCallbackHandlerWrapper(@NonNull Handler handler, @NonNull CameraCaptureCallback callback) {
        mHandler = handler;
        mCallback = callback;
    }

    @Override
    public void onCaptureCompleted(@NonNull CameraCaptureResult cameraCaptureResult) {
        mHandler.post(() -> mCallback.onCaptureCompleted(cameraCaptureResult));
    }

    @Override
    public void onCaptureFailed(@NonNull CameraCaptureFailure failure) {
        mHandler.post(() -> mCallback.onCaptureFailed(failure));
    }
}
