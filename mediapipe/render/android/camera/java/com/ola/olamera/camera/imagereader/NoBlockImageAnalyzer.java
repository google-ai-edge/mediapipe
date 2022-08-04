package com.ola.olamera.camera.imagereader;
/*
 *
 *  Creation    :  20-12-2
 *  Author      : jiaming.wjm@
 */

import android.media.Image;
import android.media.ImageReader;
import android.os.Build;

import androidx.annotation.GuardedBy;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.ola.olamera.camera.sensor.ImageRotationHelper;
import com.ola.olamera.util.Preconditions;

import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class NoBlockImageAnalyzer implements ImageReader.OnImageAvailableListener {

    private Executor mExecutor;

    private ImageAnalyzer mImageAnalyzer;

    private ImageRotationHelper mImageRotationHelper;

    public NoBlockImageAnalyzer(@NonNull Executor executor) {
        Preconditions.checkState(executor != null);
        mExecutor = executor;
    }


    public void setImageRotationHelper(@NonNull ImageRotationHelper imageRotationHelper) {
        mImageRotationHelper = imageRotationHelper;
    }

    public enum State {
        IDEAL,
        WORKING,
    }

    public NoBlockImageAnalyzer setImageAnalyzer(ImageAnalyzer imageAnalyzer) {
        mImageAnalyzer = imageAnalyzer;
        return this;
    }

    @GuardedBy("mStateLock")
    State mState = State.IDEAL;

    final Object mStateLock = new Object();


    @GuardedBy("mStateLock")
    //producer data
    private Image mCacheImage;

    @GuardedBy("mStateLock")
    //consumer data
    private Image mWaitingProcessImg;

    public boolean analyze(ImageReader reader) {
        if (reader == null) {
            return false;
        }

        //TEST
        //TEST_RESULT : acquireLatestImage和close即使在低端机器上，也是不太耗时(acquire:1ms,close:7ms)
        Image image = reader.acquireLatestImage();

        if (image == null) {
            return false;
        }

        if (mIsClose.get()) {
            image.close();
            return false;
        }

        //生产Image逻辑（后续如果复杂，将代码重构带Producer角色上）
        synchronized (mStateLock) {
            if (mState == State.IDEAL) {
                mState = State.WORKING;
                mWaitingProcessImg = image;
            } else {
                //更新最新Cache图片
                if (mCacheImage != null) {
                    mCacheImage.close();
                }
                mCacheImage = image;
            }
        }

        //有可能图片在等待执行，但是消息比较多，一直无法执行，导致将要执行的图片也cache，用于异常场景可以关闭
        mExecutor.execute(() -> {


            Image nextImg;

            //消费Image逻辑（后续如果复杂，将代码重构到Consumer角色上）
            synchronized (mStateLock) {
                nextImg = mWaitingProcessImg;
                mWaitingProcessImg = null;
            }

            if (mIsClose.get()) {
                return;
            }

            while (nextImg != null) {
                try {
                    if (mImageAnalyzer != null) {
                        mImageAnalyzer.analyze(
                                nextImg,
                                mImageRotationHelper != null ? mImageRotationHelper.getCameraSensorOrientation() : 0,
                                mImageRotationHelper != null ? mImageRotationHelper.getImageRotation() : 0
                        );
                    }
                } finally {
                    nextImg.close();
                }
                synchronized (mStateLock) {
                    //get next cache
                    nextImg = mCacheImage;
                    mCacheImage = null;
                }
            }

            synchronized (mStateLock) {
                mState = State.IDEAL;
            }
        });

        return true;
    }

    private AtomicBoolean mIsClose = new AtomicBoolean(false);

    public void close() {
        mIsClose.set(true);
        synchronized (mStateLock) {
            if (mCacheImage != null) {
                mCacheImage.close();
                mCacheImage = null;
            }
            if (mWaitingProcessImg != null) {
                mWaitingProcessImg.close();
                mWaitingProcessImg = null;
            }
        }
    }

    public void open() {
        mIsClose.set(false);
    }


    @Override
    public void onImageAvailable(ImageReader reader) {
        analyze(reader);
    }
}
