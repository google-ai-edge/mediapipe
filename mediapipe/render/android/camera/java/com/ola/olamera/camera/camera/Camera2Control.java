package com.ola.olamera.camera.camera;


import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.os.Build;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.ola.olamera.camera.session.CameraCaptureCallback;
import com.ola.olamera.camera.session.RepeatCaptureRequestConfig;
import com.ola.olamera.camera.session.config.CameraConfigUtils;
import com.ola.olamera.util.ArrayUtil;

import java.util.concurrent.Executor;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class Camera2Control {
    private final Camera2CameraImpl mCamera2Camera;


    public Camera2Control(Camera2CameraImpl camera2Camera) {
        mCamera2Camera = camera2Camera;
    }


    public void enableFlash(boolean enable, CameraCaptureCallback callback, Executor executor) {
        RepeatCaptureRequestConfig requestConfig;
        if (enable) {
            requestConfig = new RepeatCaptureRequestConfig() {
                @Override
                public void fillConfig(@NonNull Camera2Info cameraInfo, @NonNull CaptureRequest.Builder builder) {
                    CameraCharacteristics cameraCharacteristics = cameraInfo.getCameraCharacteristics();
                    //AE: 自动曝光不要设置有flash的模式
                    //When this control is used, the CaptureRequest#CONTROL_AE_MODE must be set to ON or OFF.
                    // Otherwise, the camera device auto-exposure related flash control (ON_AUTO_FLASH, ON_ALWAYS_FLASH, or ON_AUTO_FLASH_REDEYE) will override this control.

                    CameraConfigUtils.checkAndSetConfigIntValue(cameraCharacteristics, builder,
                            CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES, CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON);


                    Boolean enableFlash = cameraCharacteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
                    if (enableFlash != null && enableFlash) {
                        builder.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_TORCH);
                    }

                    builder.setTag("flash_torch");
                }

                @Override
                public CameraCaptureCallback getCallback() {
                    return callback;
                }

                @Override
                public Executor getCallbackExecutor() {
                    return executor;
                }
            };
        } else {
            requestConfig = new RepeatCaptureRequestConfig() {
                @Override
                public void fillConfig(@NonNull Camera2Info cameraInfo, @NonNull CaptureRequest.Builder builder) {
                    CameraCharacteristics cameraCharacteristics = cameraInfo.getCameraCharacteristics();
                    Boolean enableFlash = cameraCharacteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
                    if (enableFlash != null && enableFlash) {
                        builder.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF);
                    }
                    builder.setTag("flash_off");
                }

                @Override
                public CameraCaptureCallback getCallback() {
                    return callback;
                }

                @Override
                public Executor getCallbackExecutor() {
                    return executor;
                }

            };
        }

        mCamera2Camera.doRepeatingCaptureAction(requestConfig);
    }

    public Camera2CameraImpl getCamera2Camera() {
        return mCamera2Camera;
    }
}
