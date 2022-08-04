package com.ola.olamera.camerax;

import androidx.camera.core.impl.CameraCaptureResult;

/**
 * @author : yushan.lj@
 * @date : 2022/3/14
 */
public interface OnCaptureResultListener {
    void onCaptureResulted(CameraCaptureResult cameraCaptureResult);
}
