package com.ola.olamera.camera.imagereader;
/*
 *
 *  Creation    :  20-11-26
 *  Author      : jiaming.wjm@
 */

import android.media.ImageReader;
import android.os.Build;
import android.os.Handler;

import com.ola.olamera.camera.camera.CameraSurfaceHelper;
import com.ola.olamera.camera.sensor.ImageRotationHelper;
import com.ola.olamera.util.Preconditions;

import java.util.concurrent.Executor;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

/**
 * 延迟初始化话的ImageReader
 * 核心解决ImageReader的宽高只有相机支持的才能生效，而当业务使用的时候，只能知道自己预期的宽高
 */
@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class DeferrableImageReader {

    private int mExpectWidth;

    private int mExpectHeight;

    private int mActualWidth;

    private int mActualHeight;

    private final int mMaxImage;

    private int mFormat;

    private Handler mSubscriptHandler;

    private ImageReader mImageReader;

    private boolean mNonBlock = false;

    private NoBlockImageAnalyzer mNoBlockImageAnalyzer;

    private ImageRotationHelper mImageRotationHelper;

    private CameraSurfaceHelper.ISuggestionCalculation mSizeCalculation = CameraSurfaceHelper.AreaClosestSizeCalculation.getInstance();


    public static class Builder {

        private int mExpectWidth;

        private int mExpectHeight;

        private Handler mSubscriptHandler;

        private int mFormat;

        private ImageAnalyzer mImageAnalyzer;

        private Executor mHandlerExecutor;


        private CameraSurfaceHelper.ISuggestionCalculation mSizeCalculation = CameraSurfaceHelper.AreaClosestSizeCalculation.getInstance();

        public Builder setSizeCalculation(@NonNull CameraSurfaceHelper.ISuggestionCalculation sizeCalculation) {
            mSizeCalculation = sizeCalculation;
            return this;
        }

        public Builder setExpectWidth(int expectWidth) {
            mExpectWidth = expectWidth;
            return this;
        }

        public Builder setExpectHeight(int expectHeight) {
            mExpectHeight = expectHeight;
            return this;
        }


        public Builder setSubscriptHandler(Handler subscriptHandler) {
            mSubscriptHandler = subscriptHandler;
            return this;

        }

        public Builder setFormat(int format) {
            mFormat = format;
            return this;
        }

        public Builder setImageAnalyzer(ImageAnalyzer imageAnalyzer) {
            mImageAnalyzer = imageAnalyzer;
            return this;
        }


        public Builder setHandlerExecutor(Executor handlerExecutor) {
            mHandlerExecutor = handlerExecutor;
            return this;
        }

        public DeferrableImageReader build() {
            Preconditions.checkState(mExpectWidth != 0);
            Preconditions.checkState(mExpectHeight != 0);
            Preconditions.checkState(mHandlerExecutor != null);
            Preconditions.checkState(mFormat != 0);
            return new DeferrableImageReader(
                    mExpectWidth, mExpectHeight, mFormat,
                    new NoBlockImageAnalyzer(mHandlerExecutor).setImageAnalyzer(mImageAnalyzer),
                    /**
                     * 在当前{@link NoBlockImageAnalyzer}的模式下，必须是3
                     * 1,用来消费接收
                     * 2.用来缓存
                     * 3.用来消费
                     */
                    3,
                    mSubscriptHandler,
                    mSizeCalculation
            );

        }
    }

    private DeferrableImageReader(int expectWidth,
                                  int expectHeight,
                                  int format,
                                  NoBlockImageAnalyzer analyzer,
                                  int maxImage,
                                  Handler subscriptHandler,
                                  CameraSurfaceHelper.ISuggestionCalculation calculation
    ) {
        mExpectWidth = expectWidth;
        mExpectHeight = expectHeight;
        mFormat = format;
        mMaxImage = maxImage;
        mSubscriptHandler = subscriptHandler;
        mNoBlockImageAnalyzer = analyzer;
        mSizeCalculation = calculation;
    }

    public NoBlockImageAnalyzer getNoBlockImageAnalyzer() {
        return mNoBlockImageAnalyzer;
    }

    public void closePipe() {
        mNoBlockImageAnalyzer.close();
    }

    public void openPipe() {
        mNoBlockImageAnalyzer.open();
    }


    public int getFormat() {
        return mFormat;
    }

    public void createAndroidImageReader(int width, int height, ImageRotationHelper imageRotationHelper) {
        mActualWidth = width;
        mActualHeight = height;
        mImageReader = ImageReader.newInstance(width, height, mFormat, mMaxImage);
        mImageRotationHelper = imageRotationHelper;
        mNoBlockImageAnalyzer.setImageRotationHelper(imageRotationHelper);
        mImageReader.setOnImageAvailableListener(mNoBlockImageAnalyzer, mSubscriptHandler);
    }


    public ImageRotationHelper getImageRotationHelper() {
        return mImageRotationHelper;
    }

    public ImageReader unWrapper() {
        return mImageReader;
    }

    public int getExpectWidth() {
        return mExpectWidth;
    }

    public int getExpectHeight() {
        return mExpectHeight;
    }

    public int getActualWidth() {
        return mActualWidth;
    }

    public int getActualHeight() {
        return mActualHeight;
    }

    public void safeClose() {
        //TODO 目前safeClose的逻辑没有生效，后续需要在所有Image处理完成之后，调用SafeClose
        if (mImageReader != null) {
            try {
                mImageReader.close();
            } catch (Exception e) {

            }
            mImageReader = null;
        }
    }

    @NonNull
    public CameraSurfaceHelper.ISuggestionCalculation getSizeCalculation() {
        return mSizeCalculation;
    }
}
