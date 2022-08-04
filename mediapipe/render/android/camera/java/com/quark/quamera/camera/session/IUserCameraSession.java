package com.quark.quamera.camera.session;
/*
 * Copyright (C) 2005-2019 UCWeb Inc. All rights reserved.
 *  Description :
 *
 *  Creation    :  20-11-18
 *  Author      : jiaming.wjm@
 */

public interface IUserCameraSession {

    enum State {
        ACTIVE,
        INACTIVE
    }


     boolean active();

     boolean inactive();

     boolean isActive();

}
