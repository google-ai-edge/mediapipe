package com.ola.olamera.camera.session;
/*
 *
 *  Creation    :  20-12-19
 *  Author      : jiaming.wjm@
 */

import android.hardware.Camera;
import android.hardware.camera2.CaptureRequest;

import androidx.annotation.NonNull;

import com.ola.olamera.camera.camera.Camera2Info;

import java.util.concurrent.Executor;

//TODO 后面要解决fill里面的属性都不支持的时候，就不需要重新发起repeatSession
public interface RepeatCaptureRequestConfig {


    void fillConfig(@NonNull Camera2Info cameraInfo,
                    @NonNull CaptureRequest.Builder builder);

    CameraCaptureCallback getCallback();

    default Executor getCallbackExecutor() {
        return null;
    }


}
