package com.quark.quamera.camera.session;
/*
 * Copyright (C) 2005-2019 UCWeb Inc. All rights reserved.
 *  Description :
 *
 *  Creation    :  2021/5/26
 *  Author      : jiaming.wjm@alibaba-inc.com
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
