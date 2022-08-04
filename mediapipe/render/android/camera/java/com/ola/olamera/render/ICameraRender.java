package com.ola.olamera.render;

import android.opengl.EGLContext;

import com.ola.olamera.camera.camera.Camera2CameraImpl;

import java.util.Map;

public interface ICameraRender {

    boolean update(EGLContext context, int textureId, Map<String,Object> cameraInfo);

    boolean needRender();

    public interface ICameraFrameAvailableListener {
        public void onCameraFrameAvailable();
    }

    public void setCameraFrameAvailableListener(ICameraFrameAvailableListener listener);


    int[] getCameraCaptureSize();

    float[] getOESMatrix();

    boolean isMatrixInverseWidthHeight();


    void destroySurface();

    Camera2CameraImpl getCamera();
}
