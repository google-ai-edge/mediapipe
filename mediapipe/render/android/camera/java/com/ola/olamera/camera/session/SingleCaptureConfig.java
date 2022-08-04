package com.ola.olamera.camera.session;
/*
 *
 *  Creation    :  2021/6/18
 *  Author      : jiaming.wjm@
 */

import android.graphics.RectF;
import android.view.View;

import androidx.annotation.IntRange;

public class SingleCaptureConfig {

    private boolean mDetectDeviceRotation;

    private byte mJpegQuality = 100;

    public boolean useNatureRotation() {
        return mDetectDeviceRotation;
    }

    private RectF mCameraShowRect ;

    public RectF getCameraShowRect() {
        return mCameraShowRect;
    }

    public void setCameraShowRect(RectF cameraShowRect) {
        mCameraShowRect = cameraShowRect;
    }

    /**
     * 是否根据手机姿态照片旋转到正确的角度(部分手机旋转信息放在拍照出来的jpeg的exif信息中)
     */
    public SingleCaptureConfig setDetectDeviceRotation(boolean detectDeviceRotation) {
        mDetectDeviceRotation = detectDeviceRotation;
        return this;
    }

    public byte getJpegQuality() {
        return mJpegQuality;
    }

    /**
     * <p>Compression quality of the final JPEG
     * image.</p>
     * <p>85-95 is typical usage range. This tag is also used to describe the quality
     * of the HEIC image capture.</p>
     * <p><b>Range of valid values:</b><br>
     * 1-100; larger is higher quality</p>
     * <p>This key is available on all devices.</p>
     */
    public SingleCaptureConfig setJpegQuality(@IntRange(from = 0, to = 100) int jpegQuality) {
        mJpegQuality = (byte) jpegQuality;
        return this;
    }
}
