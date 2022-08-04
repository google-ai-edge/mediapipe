package com.ola.olamera.render.detector;
/*
 *
 *  Creation    :  2021/1/29
 *  Author      : jiaming.wjm@
 */

import android.content.Context;
import android.opengl.EGLContext;
import android.os.Build;

import androidx.annotation.NonNull;

import com.ola.olamera.render.entry.RenderFlowData;
import com.ola.olamera.render.expansion.IRenderExpansion;

public class AlgTextureConsumerCompat implements IAlgTextureConsumer, IRenderExpansion {

    private final IAlgTextureConsumer mConsumer;

    public AlgTextureConsumerCompat(Context context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            mConsumer = new AlgTextureConsumer(context);
        } else {
            mConsumer = null;
        }
    }

    @Override
    public void updateInputTexture(EGLContext eglContext, int textureId, int width, int height, long timeStamp) {
        if (mConsumer != null) {
            mConsumer.updateInputTexture(eglContext, textureId, width, height, timeStamp);
        }
    }

    public void release() {
        if (mConsumer != null) {
            mConsumer.release();
        }
    }

    @Override
    public void resume() {
        if (mConsumer != null) {
            mConsumer.resume();
        }
    }

    @Override
    public void pause() {
        if (mConsumer != null) {
            mConsumer.pause();
        }
    }

    @Override
    public void setOnAlgCpuDataReceiver(OnAlgCpuDataReceiver receiver) {
        if (mConsumer != null) {
            mConsumer.setOnAlgCpuDataReceiver(receiver);
        }
    }


    @NonNull
    @Override
    public RenderFlowData render(@NonNull RenderFlowData input, long timestamp) {
        return input;
    }

    @Override
    public void onSurfaceDestroy() {
        release();
    }
}
