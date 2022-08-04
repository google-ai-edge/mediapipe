package com.ola.olamera.camerax.filter;

import android.annotation.SuppressLint;

import java.util.ArrayList;
import java.util.List;

import androidx.annotation.NonNull;
import androidx.camera.core.CameraInfo;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.impl.CameraInfoInternal;
import androidx.camera.core.impl.LensFacingCameraFilter;
import androidx.core.util.Preconditions;

/**
 * @author : liujian
 * @date : 2021/7/30
 */
@SuppressLint({"UnsafeExperimentalUsageError", "RestrictedApi", "UnsafeOptInUsageError"})
public class CameraIdLensFacingCameraFilter extends LensFacingCameraFilter {

    @CameraSelector.LensFacing
    private final int mLensFacing;
    @NonNull
    private final String mCameraId;

    @SuppressLint("RestrictedApi")
    public CameraIdLensFacingCameraFilter(int lensFacing, @NonNull String cameraId) {
        super(lensFacing);
        this.mLensFacing = lensFacing;
        this.mCameraId = cameraId;
    }

    @NonNull
    @Override
    public List<CameraInfo> filter(@NonNull List<CameraInfo> cameraInfos) {
        List<CameraInfo> result = new ArrayList<>();
        for (CameraInfo cameraInfo : cameraInfos) {
            Preconditions.checkArgument(cameraInfo instanceof CameraInfoInternal,
                    "The camera info doesn't contain internal implementation.");
            Integer lensFacing = ((CameraInfoInternal) cameraInfo).getLensFacing();
            String cameraId = ((CameraInfoInternal) cameraInfo).getCameraId();
            if (lensFacing != null && lensFacing == mLensFacing && mCameraId.equals(cameraId)) {
                result.add(cameraInfo);
            }
        }

        if (result.isEmpty() && !cameraInfos.isEmpty()){
            result.add(cameraInfos.get(0));
        }
        return result;
    }
}
