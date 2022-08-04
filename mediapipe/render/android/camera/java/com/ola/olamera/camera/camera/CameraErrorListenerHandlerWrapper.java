package com.ola.olamera.camera.camera;


import android.os.Handler;

import androidx.annotation.NonNull;


public class CameraErrorListenerHandlerWrapper implements ICameraErrorListener {

    private Handler mHandler;

    private ICameraErrorListener mListener;

    public CameraErrorListenerHandlerWrapper(@NonNull Handler handler, @NonNull ICameraErrorListener listener) {
        mHandler = handler;
        mListener = listener;
    }

    @Override
    public void onError(int cameraError, String message) {
        mHandler.post(() -> mListener.onError(cameraError, message));
    }
}
