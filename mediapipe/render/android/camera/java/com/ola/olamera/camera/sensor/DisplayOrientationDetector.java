package com.ola.olamera.camera.sensor;


import android.content.Context;
import android.view.OrientationEventListener;

import com.ola.olamera.util.CameraInit;
import com.ola.olamera.util.CameraLogger;
import com.ola.olamera.util.CollectionUtil;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * TODO: 目前来看,OrientationEventListener的灵敏度不太够
 * <p>
 * TODO: 后续基于磁场，重力传感器，实现灵命的横竖屏感应
 */
public class DisplayOrientationDetector extends OrientationEventListener implements IOrientationDetector {


    private final AtomicInteger mDeviceNatureRotation = new AtomicInteger();

    private final List<WeakReference<OrientChangeListener>> mListeners;

    private final String mTag;

    private boolean enableRotate;

    public DisplayOrientationDetector(Context context, String tag) {
        super(context);
        mTag = tag;
        mListeners = new ArrayList<>();
        enableRotate = true;
    }

    /**
     * 注意是虚引用
     */
    public synchronized void addListener(OrientChangeListener listener) {
        mListeners.add(new WeakReference<>(listener));
    }

    public synchronized void removeListener(OrientChangeListener listener) {
        Iterator<WeakReference<OrientChangeListener>> it = mListeners.iterator();
        while (it.hasNext()) {
            WeakReference<OrientChangeListener> ref = it.next();
            OrientChangeListener l = ref.get();
            if (l == null) {
                it.remove();
                continue;
            }

            if (l == listener) {
                it.remove();
            }
        }
    }

    private final AtomicInteger mStartCount = new AtomicInteger();

    public void start() {
        if (super.canDetectOrientation()) {
            CameraLogger.i(mTag, "start display orientation detect");
            super.enable();
        }
    }

    public void stop() {
        CameraLogger.i(mTag, "stop display orientation detect");
        super.disable();
    }

    public void forceQuit() {
        CameraLogger.i(mTag, "force stop display orientation detect");
        super.disable();
    }

    public synchronized void clearAllListeners() {
        mListeners.clear();
    }

    @Override
    public void onOrientationChanged(int orientation) {
        if (!enableRotate) {
            return;
        }
        if (orientation == OrientationEventListener.ORIENTATION_UNKNOWN) {
            return;
        }
        int newOrientation = ((orientation + 45) / 90 * 90) % 360;
        // As "getDefaultDisplay().getRotation()" is counter-clockwise, but sensor is clockwise
        if (newOrientation == 90) {
            newOrientation = 270;
        } else if (newOrientation == 270) {
            newOrientation = 90;
        }
        if (newOrientation != mDeviceNatureRotation.get()) {
            if (CameraInit.getConfig().isDebuggable()) {
                CameraLogger.i(mTag, "display orientation changed " + newOrientation);
            }

            mDeviceNatureRotation.set(newOrientation);
            synchronized (DisplayOrientationDetector.this) {
                CollectionUtil.forEach(mListeners, ref -> {
                    OrientChangeListener l = ref.get();
                    if (l != null) {
                        l.onOrientationChanged(mDeviceNatureRotation);
                    }
                });
            }
        }
    }

    public AtomicInteger getDeviceDisplayRotation() {
        return mDeviceNatureRotation;
    }

    public void setEnableRotate(final boolean enableRotate) {
        if (!enableRotate) {
            synchronized (DisplayOrientationDetector.this) {
                CollectionUtil.forEach(mListeners, ref -> {
                    OrientChangeListener l = ref.get();
                    if (l != null) {
                        l.onOrientationChanged(new AtomicInteger(0));
                    }
                });
            }
        }
        this.enableRotate = enableRotate;
    }
}
