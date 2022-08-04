package com.ola.olamera.camera.session;
/*
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
