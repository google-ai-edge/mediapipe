package com.ola.olamera.render.detector;
/*
 *
 *  Creation    :  2021/1/29
 *  Author      : jiaming.wjm@
 */

import static com.ola.olamera.render.detector.IAlgDetector.InputDataType.NV21;

import android.content.Context;
import android.opengl.EGL14;

import androidx.annotation.NonNull;

import com.ola.olamera.render.entry.RenderFlowData;
import com.ola.olamera.render.expansion.IRenderExpansion;
import com.ola.olamera.util.CameraShould;
import com.ola.olamera.util.CollectionUtil;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class RenderExpansionManager implements IRenderExpansion {


    private final LinkedHashMap<Class<?>, IRenderExpansion> mRenderExpansions = new LinkedHashMap<>();

    private final Executor mGLExecutor;

    private final Context mContext;

    private SurfaceCreateEventHolder mSurfaceCreateEventHolder;

    private SurfaceChangeEventHolder mSurfaceChangeEventHolder;

    public synchronized <T extends IRenderExpansion> RenderExpansionManager addRenderExpansion(Class<T> clz, @NonNull T renderExpansion) {
        CameraShould.beTrue(renderExpansion != null);
        renderExpansion.setGLExecutor(mGLExecutor);
        if (mRenderExpansions.containsValue(renderExpansion)) {
            CameraShould.fail("add the same expansion multiply times");
            return this;
        }

        mRenderExpansions.put(clz, renderExpansion);

        notifyMissSurfaceEvent(renderExpansion);

        return this;
    }

    private void notifyMissSurfaceEvent(@NonNull IRenderExpansion expansion) {
        if (mSurfaceCreateEventHolder != null || mSurfaceChangeEventHolder != null) {
            mGLExecutor.execute(() -> {
                //double check in gl thread
                if (mSurfaceCreateEventHolder != null) {
                    expansion.onSurfaceCreated(mSurfaceCreateEventHolder.gl, mSurfaceCreateEventHolder.config);
                }
                //double check in gl thread
                if (mSurfaceChangeEventHolder != null) {
                    expansion.onSurfaceChanged(mSurfaceChangeEventHolder.gl, mSurfaceChangeEventHolder.width, mSurfaceChangeEventHolder.height);
                }
            });
        }
    }


    public RenderExpansionManager(Context context, Executor executor) {
        mContext = context;
        mGLExecutor = executor;
    }


    @NonNull
    @Override
    public RenderFlowData render(@NonNull RenderFlowData input, long timestamp) {
        RenderFlowData lastResult = input;
        for (IRenderExpansion expansion : mRenderExpansions.values()) {
            RenderFlowData temp = expansion.render(lastResult, timestamp);
            if (temp.texture >= 0 && lastResult != temp) {
                lastResult.recycle();
                lastResult = temp;
            }
        }
        return lastResult;
    }


    private static class SurfaceLifeEventHolder {
        GL10 gl;

        public SurfaceLifeEventHolder(GL10 gl) {
            this.gl = gl;
        }
    }

    private static class SurfaceCreateEventHolder extends SurfaceLifeEventHolder {
        EGLConfig config;

        public SurfaceCreateEventHolder(GL10 gl, EGLConfig config) {
            super(gl);
            this.config = config;
        }
    }

    private static class SurfaceChangeEventHolder extends SurfaceLifeEventHolder {
        int width;
        int height;

        public SurfaceChangeEventHolder(GL10 gl, int width, int height) {
            super(gl);
            this.width = width;
            this.height = height;
        }
    }


    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        mSurfaceCreateEventHolder = new SurfaceCreateEventHolder(gl, config);
        synchronized (mRenderExpansions) {
            CollectionUtil.forEach(mRenderExpansions.values(), expansion -> expansion.onSurfaceCreated(gl, config));
        }
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        mSurfaceChangeEventHolder = new SurfaceChangeEventHolder(gl, width, height);
        synchronized (mRenderExpansions) {
            CollectionUtil.forEach(mRenderExpansions.values(), expansion -> expansion.onSurfaceChanged(gl, width, height));
        }
    }

    @Override
    public void onSurfaceDestroy() {
        mSurfaceChangeEventHolder = null;
        mSurfaceCreateEventHolder = null;
        synchronized (mRenderExpansions) {
            CollectionUtil.forEach(mRenderExpansions.values(), IRenderExpansion::onSurfaceDestroy);
        }
    }


    public <T extends IRenderExpansion> T getRenderExpansion(Class<T> clz) {
        return (T) mRenderExpansions.get(clz);
    }
}
