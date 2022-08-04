package com.ola.olamera.camera.concurrent;
/*
 *
 *  Creation    :  2021/2/8
 *  Author      : jiaming.wjm@
 */


import android.os.Handler;

import androidx.annotation.NonNull;


import com.ola.olamera.util.CameraShould;

import java.util.concurrent.Callable;
import java.util.concurrent.Delayed;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public class HandlerScheduledExecutorService {
    private final Handler mHandler;

    public HandlerScheduledExecutorService(@NonNull Handler handler) {
        mHandler = handler;
    }


    public ScheduledFuture<?> schedule(Callable<?> task, long delay_ms) {
        HandlerScheduledFeature<?> feature = new HandlerScheduledFeature<>(mHandler, delay_ms, task);
        mHandler.postDelayed(feature, delay_ms);
        return feature;
    }

    public static class HandlerScheduledFeature<T> implements ScheduledFuture<T>, Runnable {

        private final long mRunAtMillis;
        private boolean isCanceled;
        private final Handler mHandler;
        private final Callable<?> mTask;
        private boolean isDone;

        public HandlerScheduledFeature(@NonNull Handler handler, long delayMs, @NonNull Callable<T> task) {
            mRunAtMillis = System.currentTimeMillis() + delayMs;
            mHandler = handler;
            mTask = task;
        }

        @Override
        public long getDelay(TimeUnit unit) {
            return unit.convert(mRunAtMillis - System.currentTimeMillis(),
                    TimeUnit.MILLISECONDS);
        }

        @Override
        public int compareTo(Delayed o) {
            return Long.compare(getDelay(TimeUnit.MILLISECONDS), o.getDelay(TimeUnit.MILLISECONDS));
        }

        @Override
        public boolean cancel(boolean mayInterruptIfRunning) {
            isCanceled = true;
            mHandler.removeCallbacks(this);
            return !isDone;
        }

        @Override
        public boolean isCancelled() {
            return isCanceled;
        }

        @Override
        public boolean isDone() {
            return isDone;
        }

        @Override
        public T get() throws ExecutionException, InterruptedException {
            CameraShould.fail("not support");
            return null;
        }

        @Override
        public T get(long timeout, TimeUnit unit) throws ExecutionException, InterruptedException, TimeoutException {
            CameraShould.fail("not support");
            return null;
        }

        @Override
        public void run() {
            if (isCanceled) {
                return;
            }
            try {
                mTask.call();
            } catch (Exception e) {
                CameraShould.fail();
            }
            isDone = true;
        }
    }
}
