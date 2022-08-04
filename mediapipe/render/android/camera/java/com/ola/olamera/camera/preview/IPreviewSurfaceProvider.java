package com.ola.olamera.camera.preview;


import android.view.Surface;

import com.ola.olamera.camera.camera.Camera2CameraImpl;
import com.ola.olamera.camera.session.CameraCaptureComboCallback;

import androidx.annotation.NonNull;
import androidx.camera.core.Preview;

public interface IPreviewSurfaceProvider {
    default Surface provide(@NonNull SurfaceRequest request) {
        return null;
    }

    default void onUseComplete(Surface surface) {

    }

    /**
     * 只用于CameraX，共用接口
     */
    default Preview.SurfaceProvider providerSurface() {
        return null;
    }

    class SurfaceRequest {
        public int width;
        public int height;
        public Camera2CameraImpl camera2Camera;
        public CameraCaptureComboCallback repeatCaptureCallback;

    }
}
