package com.ola.olamera.camera.session.config;
/*
 *
 *  Creation    :  2021/6/30
 *  Author      : jiaming.wjm@
 */

import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.os.Build;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.ola.olamera.util.ArrayUtil;

public class CameraConfigUtils {
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public static boolean checkAndSetConfigIntValue(@NonNull CameraCharacteristics characteristics,
                                                    @NonNull CaptureRequest.Builder builder,
                                                    @NonNull CameraCharacteristics.Key<int[]> supportListKey,
                                                    @NonNull CaptureRequest.Key<Integer> key,
                                                    @NonNull Integer expectValue) {

        int[] support_values = characteristics.get(supportListKey);
        int index = ArrayUtil.search(support_values, expectValue);
        if (index != -1) {
            builder.set(key, expectValue);
            return true;
        }
        return false;
    }
}
