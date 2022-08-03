package com.quark.quamera.camera.camera;
/*
 * Copyright (C) 2005-2019 UCWeb Inc. All rights reserved.
 *  Description :
 *
 *  Creation    :  2021/3/29
 *  Author      : jiaming.wjm@alibaba-inc.com
 */

import android.annotation.SuppressLint;
import android.content.Context;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import android.os.Handler;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.core.util.Preconditions;

import java.util.concurrent.Executor;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
class CameraManagerCompatBaseImpl implements CameraManagerCompatImpl {


    @SuppressLint("MissingPermission")
    @Override
    public void openCamera(@NonNull CameraManager cameraManager, @NonNull String cameraId, @NonNull Handler handler, @NonNull CameraDevice.StateCallback callback) throws CameraAccessException {
        cameraManager.openCamera(cameraId, callback, handler);
    }

    @NonNull
    @Override
    public CameraCharacteristics getCameraCharacteristics(@NonNull CameraManager cameraManager, @NonNull String cameraId) throws CameraAccessException {
        return cameraManager.getCameraCharacteristics(cameraId);
    }


}
