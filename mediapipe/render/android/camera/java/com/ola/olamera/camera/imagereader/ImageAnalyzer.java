package com.ola.olamera.camera.imagereader;
/*
 *
 *  Creation    :  20-12-2
 *  Author      : jiaming.wjm@
 */

import android.media.Image;

public interface ImageAnalyzer {
    public void analyze(Image image, int cameraSensorRotation, int imageRotation);
}
