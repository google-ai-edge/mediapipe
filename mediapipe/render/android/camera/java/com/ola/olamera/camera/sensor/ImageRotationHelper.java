package com.ola.olamera.camera.sensor;


import android.os.Build;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.ola.olamera.camera.camera.Camera2Info;
import com.ola.olamera.camera.session.CameraSelector;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)

public class ImageRotationHelper {
    private final DisplayOrientationDetector mDisplayOrientationDetector;
    private final CameraSelector.CameraLenFacing mLenFacing;
    private final int mCameraSensorOrientation;


    public ImageRotationHelper(@NonNull Camera2Info camera2Info, @NonNull DisplayOrientationDetector displayOrientationHelper) {
        mDisplayOrientationDetector = displayOrientationHelper;
        mLenFacing = camera2Info.getCameraLenFacing();
        mCameraSensorOrientation = camera2Info.getSensorOrientation();
    }

    /**
     * 相机传感器的旋转角度
     * 通常情况下：
     * 前置：90
     * 后置：270
     */
    public int getCameraSensorOrientation() {
        return mCameraSensorOrientation;
    }

    /**
     * 屏幕本身的旋转角度(顺时针旋转)
     * <p>
     * 0  ：竖直屏幕的自然方向
     * 90 ：
     * 180：
     * 270：
     */
    public int getDeviceRotation() {
        return mDisplayOrientationDetector.getDeviceDisplayRotation().get();
    }

    /**
     * 相机原始纹理，拍正到用户自然角度，纹理需要旋转的角度
     */
    public int getImageRotation() {
        return CameraOrientationUtil.getCameraImageRotation(
                mLenFacing,
                mCameraSensorOrientation,
                mDisplayOrientationDetector.getDeviceDisplayRotation().get()
        );
    }
}
