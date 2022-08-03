package com.quark.quamera.camera.camera;
/*
 * Copyright (C) 2005-2019 UCWeb Inc. All rights reserved.
 *  Description :
 *
 *  Creation    :  20-12-21
 *  Author      : jiaming.wjm@alibaba-inc.com
 */

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
