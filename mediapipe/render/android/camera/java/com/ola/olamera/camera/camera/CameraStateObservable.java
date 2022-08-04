package com.ola.olamera.camera.camera;
/*
 *
 *  Creation    :  2021/6/6
 *  Author      : jiaming.wjm@
 */

import java.lang.ref.SoftReference;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class CameraStateObservable {

    private final List<SoftReference<ICameraStateListener>> mListeners = new ArrayList<>();

    public void addListener(ICameraStateListener listener) {
        synchronized (mListeners) {
            Iterator<SoftReference<ICameraStateListener>> it = mListeners.iterator();
            while (it.hasNext()) {
                ICameraStateListener temp = it.next().get();
                if (temp == null) {
                    it.remove();
                    continue;
                }
                if (temp == listener) {
                    return;
                }
            }
            mListeners.add(new SoftReference<>(listener));
        }
    }

    public void clear() {
        synchronized (mListeners) {
            mListeners.clear();
        }
    }

    public void removeListener(ICameraStateListener listener) {
        synchronized (mListeners) {

            for (SoftReference<ICameraStateListener> rf : mListeners) {
                if (rf.get() == listener) {
                    mListeners.remove(rf);
                    return;
                }
            }
        }
    }

    public void notifyStateChange(CameraState to) {
        synchronized (mListeners) {
            for (SoftReference<ICameraStateListener> rf : mListeners) {
                ICameraStateListener listener = rf.get();
                if (listener != null) {
                    listener.onCameraStateChanged(to);
                }
            }
        }
    }
}
