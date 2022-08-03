package com.quark.quamera.camera.preview;
/*
 * Copyright (C) 2005-2019 UCWeb Inc. All rights reserved.
 *  Description :
 *
 *  Creation    :  20-12-18
 *  Author      : jiaming.wjm@alibaba-inc.com
 */

import android.view.Surface;

import com.quark.quamera.camera.camera.Camera2CameraImpl;
import com.quark.quamera.camera.session.CameraCaptureComboCallback;

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
