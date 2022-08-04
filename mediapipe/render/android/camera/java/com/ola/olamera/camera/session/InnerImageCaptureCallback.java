package com.ola.olamera.camera.session;
/*
 *
 *  Creation    :  2021/7/13
 *  Author      : jiaming.wjm@
 */

import android.media.Image;

public interface InnerImageCaptureCallback {


    void onCaptureStart();

    void onCaptureSuccess(Image image);

    void onError(Exception e);

}
