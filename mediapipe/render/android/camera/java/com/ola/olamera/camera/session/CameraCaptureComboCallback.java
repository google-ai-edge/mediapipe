package com.ola.olamera.camera.session;
/*
 *
 *  Creation    :  2021/5/26
 *  Author      : jiaming.wjm@
 */

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.List;

public class CameraCaptureComboCallback extends CameraCaptureCallback {
    private final List<CameraCaptureCallback> mCallbackList = new ArrayList<>();

    public void addCallback(CameraCaptureCallback callback) {
        mCallbackList.add(callback);
    }

    public void removeCallback(CameraCaptureCallback callback) {
        mCallbackList.remove(callback);
    }

    @Override
    public void onCaptureCompleted(@NonNull CameraCaptureResult cameraCaptureResult) {
        for (CameraCaptureCallback cameraCaptureCallback : mCallbackList) {
            cameraCaptureCallback.onCaptureCompleted(cameraCaptureResult);
        }
    }

    @Override
    public void onCaptureFailed(@NonNull CameraCaptureFailure failure) {
        for (CameraCaptureCallback cameraCaptureCallback : mCallbackList) {
            cameraCaptureCallback.onCaptureFailed(failure);
        }
    }
}
