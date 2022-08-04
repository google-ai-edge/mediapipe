package com.ola.olamera.camera.session;

import com.ola.olamera.camera.preview.SurfaceTextureWrapper;

public abstract class ISurfaceUpdateListener extends CameraCaptureCallback {


    private SurfaceTextureWrapper mSurfaceTextureWrapper;

    public SurfaceTextureWrapper getSurfaceTextureWrapper() {
        return mSurfaceTextureWrapper;
    }

    public void onCreate(SurfaceTextureWrapper surfaceTextureWrapper) {
        mSurfaceTextureWrapper = surfaceTextureWrapper;
        onCreate();
    }

//    @Override
//    public void onCaptureCompleted(@NonNull CameraCaptureResult cameraCaptureResult) {
//    }
//
//    @Override
//    public void onCaptureFailed(@NonNull CameraCaptureFailure failure) {
//    }


    protected abstract void onRelease();

    protected abstract void onCreate();

}