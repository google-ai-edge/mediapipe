package com.ola.olamera.render.expansion;
/*
 *
 *  Creation    :  2021/2/25
 *  Author      : jiaming.wjm@
 */

import androidx.annotation.NonNull;

import com.ola.olamera.render.entry.RenderFlowData;

import java.util.concurrent.Executor;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public interface IRenderExpansion {


    default void onSurfaceCreated(GL10 gl, EGLConfig config) {
    }

    default void onSurfaceChanged(GL10 gl, int width, int height) {

    }

    default void setGLExecutor(Executor GLExecutor) {
    }

    default Executor getGLExecutor() {
        return null;
    }


    default boolean needRender() {
        return true;
    }

    @GLThread
    @NonNull
    RenderFlowData render(@NonNull RenderFlowData input, long timestamp);


    @GLThread
    void onSurfaceDestroy();


}
