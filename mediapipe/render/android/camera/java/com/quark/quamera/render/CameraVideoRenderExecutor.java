package com.quark.quamera.render;
/*
 *
 *  Creation    :  2021/4/13
 *  Author      : jiaming.wjm@
 */

import com.quark.quamera.render.view.AndroidGLSurfaceView;

import java.util.concurrent.Executor;

public class CameraVideoRenderExecutor implements Executor {

    private AndroidGLSurfaceView mGLSurfaceView;

    public CameraVideoRenderExecutor(AndroidGLSurfaceView GLSurfaceView) {
        mGLSurfaceView = GLSurfaceView;
    }


    @Override
    public void execute(Runnable command) {
        mGLSurfaceView.queueEvent(command);
    }
}
