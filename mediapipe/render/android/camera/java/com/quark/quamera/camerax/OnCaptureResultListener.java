package com.quark.quamera.camerax;

import androidx.camera.core.impl.CameraCaptureResult;

/**
 * @author : yushan.lj@alibaba-inc.com
 * @date : 2022/3/14
 */
public interface OnCaptureResultListener {
    void onCaptureResulted(CameraCaptureResult cameraCaptureResult);
}
