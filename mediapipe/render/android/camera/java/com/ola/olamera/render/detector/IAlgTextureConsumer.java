package com.ola.olamera.render.detector;
/*
 *
 *  Creation    :  2021/1/29
 *  Author      : jiaming.wjm@
 */

import android.opengl.EGLContext;

import java.util.concurrent.locks.ReentrantReadWriteLock;

public interface IAlgTextureConsumer {

    void updateInputTexture(EGLContext eglContext, int textureId, int width, int height, long timeStamp);

    void release();

    void resume();

    void pause();

    class NV21Buffer {

        private final byte[] mBytes;
        private final int mWidth;
        private final int mHeight;
        private final ReentrantReadWriteLock mLock = new ReentrantReadWriteLock();

        public NV21Buffer(int width, int height) {
            this.mWidth = width;
            this.mHeight = height;
            int yuv_size = (int) (width * (((float) height) * 3 / 2));
            mBytes = new byte[yuv_size];
        }

        public int getWidth() {
            return mWidth;
        }

        public int getHeight() {
            return mHeight;
        }

        public byte[] readLock() {
            mLock.readLock().lock();
            return mBytes;
        }

        public byte[] writeLock() {
            mLock.writeLock().lock();
            return mBytes;
        }

        public void writeUnlock() {
            mLock.writeLock().unlock();
        }


        public void readUnlock() {
            mLock.readLock().unlock();
        }


    }

    interface OnAlgCpuDataReceiver {
        void onReceiveCpuData(NV21Buffer buffer, long timestamp);
    }

    void setOnAlgCpuDataReceiver(OnAlgCpuDataReceiver receiver);

}
