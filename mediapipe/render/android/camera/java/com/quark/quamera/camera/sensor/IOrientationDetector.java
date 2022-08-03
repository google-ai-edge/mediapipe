package com.quark.quamera.camera.sensor;
/*
 * Copyright (C) 2005-2019 UCWeb Inc. All rights reserved.
 *  Description :
 *
 *  Creation    :  20-12-14
 *  Author      : jiaming.wjm@alibaba-inc.com
 */

import java.util.concurrent.atomic.AtomicInteger;

public interface IOrientationDetector {
    public void addListener(OrientChangeListener listener);

    public void removeListener(OrientChangeListener listener);

    interface OrientChangeListener {
        public void onOrientationChanged(AtomicInteger orientation);
    }
}

