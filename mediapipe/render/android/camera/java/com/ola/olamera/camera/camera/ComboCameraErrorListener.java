package com.ola.olamera.camera.camera;


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
