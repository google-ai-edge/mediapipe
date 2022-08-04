package com.ola.olamera.camera.session;


import androidx.annotation.NonNull;

import java.util.concurrent.Executor;

public class SurfaceUpdateListenerExecutorWrapper extends ISurfaceUpdateListener {

    private Executor mExecutor;
    private ISurfaceUpdateListener mListener;


    public SurfaceUpdateListenerExecutorWrapper(@NonNull Executor executor, @NonNull ISurfaceUpdateListener listener) {
        mExecutor = executor;
        mListener = listener;
    }

    @Override
    public void onRelease() {
        mExecutor.execute(() -> mListener.onRelease());
    }

    @Override
    public void onCreate() {
        mExecutor.execute(() -> mListener.onCreate());
    }
}
