package com.ola.olamera.camera.session;
/*
 *
 *  Creation    :  20-11-18
 *  Author      : jiaming.wjm@
 */

import java.util.List;

public interface ISelector {
    public List<String> filter(List<String> cameraIds);
}
