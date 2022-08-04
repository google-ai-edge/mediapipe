package com.ola.olamera.render.detector;
/*
 *
 *  Creation    :  2021/1/29
 *  Author      : jiaming.wjm@
 */

import androidx.annotation.GuardedBy;

public abstract class BaseAlgDetector<T> implements IAlgDetector<T> {


    // @GuardedBy("mStateLock")
    protected volatile State mState = State.UNINITIALIZED;

    @Override
    public void init() {
        if (onInit()) {
            mState = State.INITIALIZED;
        } else {
            mState = State.UNINITIALIZED;
        }
    }

    public abstract boolean onInit();

    public abstract void onRelease();

    public abstract boolean onStart();

    public abstract void onStop();

    public abstract T detectInner(IAlgTextureConsumer.NV21Buffer buffer, long timestamp);


    @Override
    public T detect(IAlgTextureConsumer.NV21Buffer buffer, long timestamp) {
        if (mState == State.RUNNING) {
            return detectInner(buffer, timestamp);
        }
        return null;
    }

    @Override
    public void release() {
        onRelease();
    }

    @Override
    public void start() {
        if (mState == State.INITIALIZED) {
            if (onStart()) {
                mState = State.RUNNING;
            }
        }
    }

    @Override
    public void stop() {
        if (mState == State.RUNNING) {
            mState = State.INITIALIZED;
            onStop();
        }
    }

    @Override
    public State getState() {
        return mState;
    }
}
