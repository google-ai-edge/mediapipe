package com.ola.olamera.camera.sensor;


import java.util.concurrent.atomic.AtomicInteger;

public interface IOrientationDetector {
    public void addListener(OrientChangeListener listener);

    public void removeListener(OrientChangeListener listener);

    interface OrientChangeListener {
        public void onOrientationChanged(AtomicInteger orientation);
    }
}

