package com.ola.olamera.camera.preview;
/*
 *
 *  Creation    :  2020/4/15
 *  Author      : jiaming.wjm@
 */

import android.graphics.SurfaceTexture;
import android.util.Log;
import android.view.Surface;

import com.ola.olamera.util.CameraLogger;

import java.util.Arrays;

public class SurfaceTextureWrapper {
    private SurfaceTexture mSurfaceTexture;
    private Surface mSurface;

    private volatile boolean mHasAttach;

    private volatile boolean mHasRelease;

    private int mAttachContextId = -1;

    private int[] mSize = new int[2];


    public synchronized void release() {
        try {
            if (mSurfaceTexture != null) {
                mSurfaceTexture.release();
                CameraLogger.i("SurfaceTexture", "release texture (%s)", this);
            }
        } catch (Exception ignore) {
        } finally {
            mHasRelease = true;
        }
    }


    public SurfaceTextureWrapper(int width, int height) {
        mSize[0] = width;
        mSize[1] = height;
        mSurfaceTexture = new SurfaceTexture(0);
        mSurfaceTexture.setDefaultBufferSize(width, height);
        mSurfaceTexture.detachFromGLContext();
        mSurface = new Surface(mSurfaceTexture);
    }

    public synchronized boolean isValid() {
        return !mHasRelease;
    }

    public synchronized SurfaceTexture getSurfaceTexture() {
        return mSurfaceTexture;
    }

    public synchronized boolean updateTexImage() {
        if (mHasRelease) {
            return false;
        }
        mSurfaceTexture.updateTexImage();
        return true;
    }

    public synchronized void getTransformMatrix(float[] mat) {
        if (mHasRelease) {
            return;
        }
        mSurfaceTexture.getTransformMatrix(mat);
    }

    public synchronized boolean attachToGLContext(int context_id, int attach_texture_id) {
        if (mHasRelease) {
            return false;
        }
        if (mHasAttach && mAttachContextId == context_id) {
            return true;
        }

        if (mHasAttach) {
            mSurfaceTexture.detachFromGLContext();
        }

        long start = System.nanoTime();
        mSurfaceTexture.attachToGLContext(attach_texture_id);


        CameraLogger.testLongLog("SurfaceTexture", "attachToGLContext texture:%d,  (use:%d) Object:%s ",
                attach_texture_id, (System.nanoTime() - start) / 1000000, this.toString());

        mHasAttach = true;
        mAttachContextId = context_id;
        return true;
    }

    public synchronized void detachFromGLContext(int context_id) {
        if (mHasRelease) {
            return;
        }
        if (mHasAttach && mAttachContextId == context_id) {
            long start = System.nanoTime();
            mSurfaceTexture.detachFromGLContext();

            CameraLogger.testLongLog("SurfaceTexture", "detachFromGLContext (use:%d) Object:%s ",
                    (System.nanoTime() - start) / 1000000, this.toString());

            mAttachContextId = -1;
            mHasAttach = false;
        }
    }

    public Surface getSurface() {
        return mSurface;
    }

    public int[] getSize() {
        return mSize;
    }


    @Override
    public String toString() {
        return "SurfaceTextureWrapper{" +
                "mSurfaceTexture=" + mSurfaceTexture +
                ", mHasAttach=" + mHasAttach +
                ", mHasRelease=" + mHasRelease +
                ", mAttachContextId=" + mAttachContextId +
                ", mSize=" + Arrays.toString(mSize) +
                '}';
    }
}
