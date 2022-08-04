package com.ola.olamera.camerax;
/*
 *
 *  Creation    :  2021/11/15
 *  Author      : jiaming.wjm@
 */

import android.annotation.SuppressLint;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CaptureRequest;
import android.os.Build;
import android.util.Range;
import android.util.Rational;

import java.util.concurrent.ConcurrentHashMap;

import androidx.annotation.RequiresApi;
import androidx.camera.camera2.impl.Camera2ImplConfig;
import androidx.camera.camera2.internal.Camera2CameraControlImpl;
import androidx.camera.camera2.internal.Camera2CameraInfoImpl;
import androidx.camera.core.CameraControl;
import androidx.camera.core.CameraSelector;

/**
 *
 //    private FaceMode mFaceMode = new FaceMode();

 //    @SuppressLint("RestrictedApi")
 //    private void checkFaceModeCharacteristics(CameraInfo cameraInfo) {
 //        if (!(cameraInfo instanceof Camera2CameraInfoImpl)) {
 //            return;
 //        }
 //        String id = ((Camera2CameraInfoImpl) cameraInfo).getCameraId();
 //
 //        mFaceMode.checkCamera(id, (Camera2CameraInfoImpl) cameraInfo);
 //        mCameraController.getCameraControlLiveData().observe(this, cameraControl -> mFaceMode.fillCameraCharacteristics(id, cameraControl));
 //    }

 * 暂时没用，设计目的是在不同mode下调节相机参数
 */
@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
class FaceMode {

    private static class FaceCameraConfig {
        Range<Integer> aeRange;
        Rational aeStep;


        boolean isValid() {
            return aeRange != null && aeStep != null;
        }
    }

    private ConcurrentHashMap<String, FaceCameraConfig> mCameraFaceConfig = new ConcurrentHashMap<>();


    @SuppressLint("RestrictedApi")
    public boolean checkCamera(String id, Camera2CameraInfoImpl cameraInfo) {
        if ((cameraInfo == null)) {
            return false;
        }


        if (mCameraFaceConfig.containsKey(id)) {
            return true;
        }

        //TODO 确认是否需要用这个方式判断前后摄象机(目前写死只有前置，后续可以作为一个参数填写进来)
        if (cameraInfo.getLensFacing() == null ||
                cameraInfo.getLensFacing() != CameraSelector.LENS_FACING_FRONT) {
            return false;
        }

        FaceCameraConfig config = new FaceCameraConfig();

        CameraCharacteristics cc = (cameraInfo)
                .getCameraCharacteristicsCompat().toCameraCharacteristics();
        Range<Integer> range = cc.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE);
        Rational aeStep = cc.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_STEP);
        if (range.getUpper() > 0) {
            config.aeRange = range;
            config.aeStep = aeStep;
        }

        mCameraFaceConfig.put(cameraInfo.getCameraId(), config);

        return true;
    }

    /**
     * 优先使用逻辑广角镜头
     */
    @SuppressLint({"RestrictedApi", "UnsafeOptInUsageError", "UnsafeExperimentalUsageError"})
    public void fillCameraCharacteristics(String cameraId, CameraControl cameraControl) {
        if (!mCameraFaceConfig.containsKey(cameraId)) {
            return;
        }
        FaceCameraConfig config = mCameraFaceConfig.get(cameraId);

        if (!config.isValid()) {
            return;
        }

        if (cameraControl instanceof Camera2CameraControlImpl) {
            int stepValue = (int) (1f / config.aeStep.floatValue());
            //白平衡增加一个亮度
            if (!config.aeRange.contains(stepValue)) {
                return;
            }

            Camera2ImplConfig captureRequestOption = new Camera2ImplConfig.Builder()
                    .setCaptureRequestOption(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, stepValue).build();
            Camera2CameraControlImpl controlImpl = (Camera2CameraControlImpl) cameraControl;
            controlImpl.getCamera2CameraControl().
                    addCaptureRequestOptions(captureRequestOption);
        }
    }

}
