package com.ola.olamera.camera.camera;
/*
 *
 *  Creation    :  2021/3/29
 *  Author      : jiaming.wjm@
 */

import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import android.os.Handler;

import androidx.annotation.NonNull;

import java.util.concurrent.Executor;

interface CameraManagerCompatImpl {

    /**
     * Returns a {@link CameraManagerCompatImpl} instance depending on the API level
     */
    @NonNull
    static CameraManagerCompatImpl from() {
        if (Build.VERSION.SDK_INT == 28) {
            // Can use Executor directly on API 28+
            return new CameraManagerCompatApi28Impl();
        }
        // Pass compat handler to implementation.
        return new CameraManagerCompatBaseImpl();
    }

    public void openCamera(@NonNull CameraManager cameraManager,
                           @NonNull String cameraId,
                           @NonNull Handler executor,
                           @NonNull CameraDevice.StateCallback callback) throws CameraAccessException;

    @NonNull
    public CameraCharacteristics getCameraCharacteristics(@NonNull CameraManager cameraManager, @NonNull String cameraId) throws CameraAccessException;
}
