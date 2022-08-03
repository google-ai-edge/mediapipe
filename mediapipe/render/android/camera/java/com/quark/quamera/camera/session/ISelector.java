package com.quark.quamera.camera.session;
/*
 * Copyright (C) 2005-2019 UCWeb Inc. All rights reserved.
 *  Description :
 *
 *  Creation    :  20-11-18
 *  Author      : jiaming.wjm@alibaba-inc.com
 */

import java.util.List;

public interface ISelector {
    public List<String> filter(List<String> cameraIds);
}
