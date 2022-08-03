package com.quark.quamera.camera.session;
/*
 * Copyright (C) 2005-2019 UCWeb Inc. All rights reserved.
 *  Description :
 *
 *  Creation    :  2021/7/13
 *  Author      : jiaming.wjm@alibaba-inc.com
 */

import android.media.Image;

public interface InnerImageCaptureCallback {


    void onCaptureStart();

    void onCaptureSuccess(Image image);

    void onError(Exception e);

}
