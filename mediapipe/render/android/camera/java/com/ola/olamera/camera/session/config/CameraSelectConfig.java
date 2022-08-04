package com.ola.olamera.camera.session.config;
/*
 *
 *  Creation    :  2021/5/7
 *  Author      : jiaming.wjm@
 */

import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CaptureRequest;
import android.os.Build;

import com.ola.olamera.camera.camera.Camera2Info;
import com.ola.olamera.camera.session.CameraSelector;
import com.ola.olamera.util.CameraLogger;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)

public class CameraSelectConfig {

    private final @NonNull
    CameraSelector mCameraSelector;

    public CameraSelectConfig(@NonNull CameraSelector cameraSelector) {
        mCameraSelector = cameraSelector;
    }

    public void fillConfig(@NonNull Camera2Info cameraInfo,
                           @NonNull CaptureRequest.Builder builder) {


        CameraCharacteristics cameraCharacteristics = cameraInfo.getCameraCharacteristics();

        Camera2Info.FocalLengthInfo minFocalInfo = cameraInfo.getBestFocalInfo();


        if (mCameraSelector.isUseWideCamera()) {
            //            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            //                Range<Float> zoom_range = cameraCharacteristics.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE);
            //
            //            } else
            if (minFocalInfo != null) {
                float focal_length = minFocalInfo.focalLength;
                builder.set(CaptureRequest.LENS_FOCAL_LENGTH, focal_length);
                CameraLogger.i("CameraLifeManager", "camera (%s) set focal_length %f", cameraInfo.getCameraId(), focal_length);
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                //              int[]   cameraCharacteristics.get(CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES);
            }
            //            builder.set(CaptureRequest.SCALER_CROP_REGION,);


        }
    }


}
