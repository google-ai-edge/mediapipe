/*
 * Copyright 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ola.olamera.camerax.utils;

import android.os.Handler;
import android.os.HandlerThread;

import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;

import androidx.annotation.NonNull;

public final class SingleThreadHandlerExecutor implements Executor {

    private final String mThreadName;
    private final HandlerThread mHandlerThread;
    private final Handler mHandler;

    public SingleThreadHandlerExecutor(@NonNull String threadName, int priority) {
        this.mThreadName = threadName;
        mHandlerThread = new HandlerThread(threadName, priority);
        mHandlerThread.start();
        mHandler = new Handler(mHandlerThread.getLooper());
    }

    @NonNull
    public Handler getHandler() {
        return mHandler;
    }

    @Override
    public void execute(@NonNull Runnable command) {
        if (!mHandler.post(command)) {
            throw new RejectedExecutionException(mThreadName + " is shutting down.");
        }
    }

    public boolean shutdown() {
        return mHandlerThread.quitSafely();
    }
}
