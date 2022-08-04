package com.ola.olamera.util;
/*
 *
 *  Creation    :  20-11-13
 *  Author      : jiaming.wjm@
 */

import android.os.Handler;
import android.os.HandlerThread;

import androidx.annotation.RestrictTo;

import java.util.concurrent.Executor;

//@RestrictTo(RestrictTo.Scope.LIBRARY_GROUP)
public class SequentialExecutor implements Executor {


    private HandlerThread mHandlerThread;
    private Handler mHandler;
    private final String mName;

    public SequentialExecutor(String name) {
        mName = name;
    }

    public synchronized void start() {
        mHandlerThread = new HandlerThread(mName);
        mHandlerThread.start();
        mHandler = new Handler(mHandlerThread.getLooper());
    }

    public synchronized void stop() {
        if (mHandlerThread != null) {
            mHandlerThread.quit();
            mHandlerThread = null;
            mHandler = null;
        }
    }

    public Handler getHandler() {
        if (mHandler == null) {
            throw new RuntimeException("Start First");
        }
        return mHandler;
    }

    @Override
    public synchronized void execute(Runnable command) {
        if (mHandler == null) {
            throw new RuntimeException("Executor （" + mName + ") Start First");
        }
        mHandler.post(command);
    }
}
