package com.quark.quamera.camera.camera;
/*
 * Copyright (C) 2005-2019 UCWeb Inc. All rights reserved.
 *  Description :
 *
 *  Creation    :  20-12-21
 *  Author      : jiaming.wjm@alibaba-inc.com
 */

import java.util.List;

public class ComboCameraErrorListener implements ICameraErrorListener {

    private List<ICameraErrorListener> mListeners;

    public ComboCameraErrorListener(List<ICameraErrorListener> listeners) {
        mListeners = listeners;
    }


    @Override
    public void onError(int cameraError, String message) {
        if (mListeners == null) {
            return;
        }
        for (ICameraErrorListener listener : mListeners) {
            if (listener == null) {
                continue;
            }
            listener.onError(cameraError, message);
        }
    }
}
