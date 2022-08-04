package com.quark.quamera.camera.imagereader;
/*
 * Copyright (C) 2005-2019 UCWeb Inc. All rights reserved.
 *  Description :
 *
 *  Creation    :  20-12-2
 *  Author      : jiaming.wjm@
 */

import android.media.Image;

public interface ImageAnalyzer {
    public void analyze(Image image, int cameraSensorRotation, int imageRotation);
}
