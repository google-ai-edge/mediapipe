package com.ola.olamera.render;

import android.annotation.TargetApi;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.opengl.EGL14;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;
import android.util.Rational;

import com.ola.olamera.camera.preview.ViewPort;
import com.ola.olamera.render.entry.FrameDetectData;
import com.ola.olamera.render.entry.RenderFlowData;
import com.ola.olamera.render.expansion.IRenderExpansion;
import com.ola.olamera.render.photo.ExportPhoto;
import com.ola.olamera.render.photo.SnapShotCommand;
import com.ola.olamera.render.view.AndroidGLSurfaceView;
import com.ola.olamera.util.CameraInit;
import com.ola.olamera.util.CameraLogger;
import com.ola.olamera.util.CameraShould;
import com.ola.olamera.util.ImageUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.RejectedExecutionException;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import androidx.annotation.FloatRange;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

/**
 * 渲染到屏幕，并且进行录制
 */
@TargetApi(18)
public class CameraVideoRenderPipe implements AndroidGLSurfaceView.Renderer, ICameraRender.ICameraFrameAvailableListener {

    private static final String TAG = "CameraVideoRender";

    private ScreenRenderFilter mScreenRender;//绘制到屏幕

    private GlFboFilter mCameraFilter;//直接从外部纹理绘制内容到FBO

    private @ViewPort.ScaleType
    int mCameraScaleType = ViewPort.FILL_CENTER;

    private CropFboFilter mCropFboFilter;

    private CropFboFilter mTransformFilter;

    @Deprecated
    //后续用mCameraShowRect来代替
    private Rational mCameraShowRational;

    private ICameraShowViewChangeListener mLayoutChangeListener;

    private volatile boolean mLayoutChange = false;

    private IRenderExpansion mRenderExpansion;

    private boolean mHasRelease = false;


    private final RectF mTempRectF = new RectF();
    private final RectF mCameraShowRect = new RectF();

    private int mSurfaceTextureId = -1;
    private int mWindowWidth;
    private int mWindowHeight;

    private final Context mContext;
    private SnapShotCommand mSnapShotCommand;
    private final AndroidGLSurfaceView mHost;

    public CameraVideoRenderPipe(@NonNull Context context,
                                 @NonNull AndroidGLSurfaceView host,
                                 @NonNull IRenderExpansion expansion) {
        mContext = context;
        mHost = host;
        mRenderExpansion = expansion;
    }

    public void setCameraShowLayoutChangeListener(ICameraShowViewChangeListener layoutChangeListener) {
        mLayoutChangeListener = layoutChangeListener;
    }


    public void snapshot(final SnapShotCommand snapShotCommand) {
        mSnapShotCommand = snapShotCommand;
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        initSurfaceTexture();

        CameraLogger.i("CameraVideoView", "on surfaceCreated");

        if (mScreenRender == null) {
            mScreenRender = new ScreenRenderFilter();
            mScreenRender.setScaleType();
            mScreenRender.prepare();
        }

        if (mCropFboFilter == null) {
            mCropFboFilter = new CropFboFilter();
        }

        if (mTransformFilter == null) {
            mTransformFilter = new CropFboFilter();
        }

        if (mRenderExpansion != null) {
            mRenderExpansion.onSurfaceCreated(gl, config);
        }
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        CameraLogger.i(GLConstant.TAG, "Render surface changed width=" + width + ", height=" + height + " GLContext " + EGL14.eglGetCurrentContext());


        if (mWindowWidth != width || mWindowHeight != height) {
            mLayoutChange = true;
        }

        mWindowWidth = width;
        mWindowHeight = height;

        if (mCameraShowRect.isEmpty()) {
            mCameraShowRect.set(0, 0, mWindowWidth, mWindowHeight);
        }


        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            calculateCameraShowRational();
        }

        if (mRenderExpansion != null) {
            mRenderExpansion.onSurfaceChanged(gl, width, height);
        }
    }


    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private void calculateCameraShowRational() {
        if (mWindowHeight != 0 && mWindowWidth != 0) {
            float heightPercentage = 1 - mMarginPercentage[1] - mMarginPercentage[3];
            switch (mCameraScaleType) {
                case ViewPort.FILL_CENTER:
                    mCameraShowRational = new Rational(mWindowWidth, (int) (mWindowHeight * heightPercentage));
                    break;
                case ViewPort.CENTER_INSIDE:
                    //没有相机的时候，直接提供屏幕的宽高比
                    if (mCameraSurfaceRender == null) {
                        mCameraShowRational = new Rational(mWindowWidth, (int) (mWindowHeight * heightPercentage));
                        return;
                    }
                    int[] size = mCameraSurfaceRender.getCameraCaptureSize();
                    if (mCameraSurfaceRender.isMatrixInverseWidthHeight()) {
                        mCameraShowRational = new Rational(size[1], size[0]);
                    } else {
                        mCameraShowRational = new Rational(size[0], size[1]);
                    }
                    break;
                default:
                    CameraShould.fail("not support now");
            }

        }

    }

    public Rational getCameraShowRational() {
        return mCameraShowRational;
    }

    public int getWindowWidth() {
        return mWindowWidth;
    }

    public int getWindowHeight() {
        return mWindowHeight;
    }

    public void setWindowHeight(int windowHeight) {
        mWindowHeight = windowHeight;
    }

    public void onSurfaceDestroy() {
        CameraLogger.i("CameraVideoView", "onSurfaceDestroy");

        mCameraShowRect.setEmpty();


        if (mRenderExpansion != null) {
            mRenderExpansion.onSurfaceDestroy();
            mRenderExpansion = null;
        }

        //退出的时候，确保所有任务都可以执行到（此处先忽略锁）,
        // 顺序放在RenderExpansion后面，因为其他为内部释放，不会触发Task
        //TODO 不能继续执行task任务，因为这些任务可能依赖GL环境，但是destory的时候，GL环境是不稳定的，目前GL Surface方案无法保证声明周期可控
//        while (mTasks.size() > 0) {
//            runExecutorTask();
//        }
        synchronized (mTasks) {
            mTasks.clear();
        }

        mHasRelease = true;

        if (mCameraFilter != null) {
            mCameraFilter.release();
            mCameraFilter = null;
        }

        if (mCropFboFilter != null) {
            mCropFboFilter.release();
            mCropFboFilter = null;
        }

        if (mTransformFilter != null) {
            mTransformFilter.release();
            mTransformFilter = null;
        }

        if (mScreenRender != null) {
            mScreenRender.release();
            mScreenRender = null;
        }


    }


    private void createCameraRenderFBOIfNeed() {
        int[] size = mCameraSurfaceRender.getCameraCaptureSize();

        int outWith = size[0];
        int outHeight = size[1];
        if (mCameraSurfaceRender.isMatrixInverseWidthHeight()) {
            outWith = size[1];
            outHeight = size[0];
        }
        if (mCameraFilter != null
                //size not change
                && (mCameraFilter.getOutputHeight() == outHeight
                && mCameraFilter.getOutputWidth() == outWith)) {
            return;
        }
        mCameraFilter = new GlFboFilter();
        mCameraFilter.setInputTextureId(mSurfaceTextureId);
        mCameraFilter.setInputSize(size[0], size[1]);
        mCameraFilter.setOutputSize(outWith, outHeight);
        mCameraFilter.prepare();
    }

    private void runExecutorTask() {
        try {
            List<Runnable> tasks;
            synchronized (mTasks) {
                tasks = new ArrayList<>(mTasks);
                mTasks.clear();
            }

            for (Runnable task : tasks) {
                task.run();
            }
        } catch (Exception ignore) {
        }
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        runExecutorTask();

        SnapShotCommand tempSnapshotCommand;

        //局部持有callback，避免没有使用的call 在 draw完后直接给清空了
        synchronized (CameraVideoRenderPipe.this) {
            tempSnapshotCommand = mSnapShotCommand;
        }

        try {
            renderInner(tempSnapshotCommand);
        } finally {
            if (tempSnapshotCommand != null) {
                synchronized (CameraVideoRenderPipe.this) {
                    if (tempSnapshotCommand == mSnapShotCommand) {
                        mSnapShotCommand = null;
                    }
                }
            }
        }
    }

    private final List<Runnable> mTasks = new ArrayList<>();


    public boolean queueTask(Runnable runnable) {
        if (mHasRelease) {
            throw new RejectedExecutionException("gl render thread has shutdown");
        }
        synchronized (mTasks) {
            mTasks.add(runnable);
        }
        return true;
    }


    private long mFpsCount;
    private long mResetRecordFpsTime = 0;


    //left top right bottom (margin)
    private final float[] mMarginPercentage = new float[4];

    public void setCameraShowRect(@FloatRange(from = 0.0, to = 1.0) float topMargin, @FloatRange(from = 0.0, to = 1.0) float bottomMargin) {
        if (mMarginPercentage[1] != topMargin || mMarginPercentage[3] != bottomMargin) {
            mLayoutChange = true;
        }
        mMarginPercentage[1] = topMargin;
        mMarginPercentage[3] = bottomMargin;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            calculateCameraShowRational();
        }
        requestRenderIfNeed();
    }


    private void requestRenderIfNeed() {
        if (mHost.getRenderMode() == AndroidGLSurfaceView.RENDERMODE_CONTINUOUSLY) {
            return;
        }
        if (mLayoutChange) {
            mLayoutChange = false;
            mHost.requestRender();
        }
    }


    private float getHeightPercentage() {
        return 1 - mMarginPercentage[1] - mMarginPercentage[3];
    }

    @NonNull
    public float[] getMarginPercentage() {
        return mMarginPercentage;
    }

    //TODO 后续需要实现显存服用(Texture的服用)
    private void renderInner(SnapShotCommand snapShotCommand) {
        //因为View的创建渲染早于camera打开，所以使用新的时间戳来作为整体时间戳，而不是使用相机
        long timestamp = SystemClock.uptimeMillis();

        long start = SystemClock.elapsedRealtimeNanos();

        GLES20.glEnable(GLES20.GL_BLEND);
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);


        RenderFlowData data = RenderFlowData.obtain(mWindowWidth, mWindowHeight, null);

        long step1 = SystemClock.elapsedRealtimeNanos();
        if (mCameraSurfaceRender != null && mCameraSurfaceRender.needRender()) {
            //更新相机纹理
            mCameraSurfaceRender.update(EGL14.eglGetCurrentContext(), mSurfaceTextureId, data.extraData);

            createCameraRenderFBOIfNeed();
            mCameraFilter.draw(mCameraSurfaceRender.getOESMatrix());
            data.textureWidth = mCameraFilter.getOutputWidth();
            data.textureHeight = mCameraFilter.getOutputHeight();
            data.texture = mCameraFilter.getOutputTextureId();
        }

        step1 = SystemClock.elapsedRealtimeNanos() - step1;

        long step2 = SystemClock.elapsedRealtimeNanos();

        RenderFlowData snapshotData = null;

        if (mCropFboFilter != null && data.texture != -1) {

            int outHeight = (int) (mWindowHeight * getHeightPercentage());
            //宽度发生了改变
            if (mCropFboFilter.isPrepared()) {
                if (mCropFboFilter.getOutputHeight() != outHeight) {
                    mCropFboFilter.release();
                    mCropFboFilter = new CropFboFilter();
                }
            }
            mCropFboFilter.setOutputSize(data.windowWidth, outHeight);
            mCropFboFilter.setInputTextureId(data.texture);
            mCropFboFilter.setInputSize(data.textureWidth, data.textureHeight);
            // input为相机预览的分辨率  out为输出展示的宽高  矩阵缩放
            ImageUtils.getScaleVertexMatrix(mCameraScaleType, mCropFboFilter.getPosMtx(),
                    mCropFboFilter.getInputWidth(), mCropFboFilter.getInputHeight(),
                    mCropFboFilter.getOutputWidth(), mCropFboFilter.getOutputHeight());
            mCropFboFilter.prepare();
            mCropFboFilter.draw(null);

            data = RenderFlowData.obtain(data.windowWidth, data.windowHeight, data.extraData);
            data.textureWidth = mCropFboFilter.getOutputWidth();
            data.textureHeight = mCropFboFilter.getOutputHeight();
            data.texture = mCropFboFilter.getOutputTextureId();

            snapshotData = data;
            if (mTransformFilter != null) {
                mTransformFilter.setOutputSize(data.windowWidth, mWindowHeight);
                mTransformFilter.setInputTextureId(data.texture);
                mTransformFilter.setInputSize(data.textureWidth, data.textureHeight);
                ImageUtils.getClipVertexMatrix(mTransformFilter.getPosMtx(), mMarginPercentage);
                mTransformFilter.prepare();
                mTransformFilter.draw(null);

                data = RenderFlowData.obtain(data.windowWidth, data.windowHeight, data.extraData);
                data.textureWidth = mTransformFilter.getOutputWidth();
                data.textureHeight = mTransformFilter.getOutputHeight();
                data.texture = mTransformFilter.getOutputTextureId();
            }

            calculateCameraShowLayout();
        }

        if (snapshotData == null) {
            snapshotData = data;
        }


        step2 = SystemClock.elapsedRealtimeNanos() - step2;

        long step3 = SystemClock.elapsedRealtimeNanos();

        if (mRenderExpansion != null && data.texture != -1) {
            data = mRenderExpansion.render(data, timestamp);
        }

        step3 = SystemClock.elapsedRealtimeNanos() - step3;

        long step4 = SystemClock.elapsedRealtimeNanos();


        if (snapShotCommand != null) {
            SnapshotFilter snapshotFilter = new SnapshotFilter();
            snapshotFilter.setInputSize(snapshotData.textureWidth, snapshotData.textureHeight);
            snapshotFilter.setInputTextureId(snapshotData.texture);
            snapshotFilter.prepare();
            Bitmap result = snapshotFilter.snapshot(snapShotCommand);
            snapshotFilter.release();

            FrameDetectData detectData = data.detectData;


            ExportPhoto mainExport = ExportPhoto.ofBitmap(result);

            if (mainExport != null) {
                mainExport.setHasClip(snapShotCommand.needClip())
                        .setFrameDetectData(detectData)
                        .setSnapShotCommand(snapShotCommand);

                for (SnapShotCommand command : snapShotCommand.getSubCommand()) {

                    long startTimes = SystemClock.uptimeMillis();
                    int outWidth = (int) (command.getScale() * snapshotData.textureWidth);
                    int outHeight = (int) (command.getScale() * snapshotData.textureHeight);
                    SnapshotFilter subFilter = new SnapshotFilter();
                    subFilter.setInputSize(snapshotData.textureWidth, snapshotData.textureHeight);
                    subFilter.setInputTextureId(snapshotData.texture);
                    subFilter.setOutputSize(outWidth, outHeight);
                    subFilter.prepare();

                    Bitmap subBitmap = subFilter.snapshot(snapShotCommand);
                    ExportPhoto subPhoto = ExportPhoto.ofBitmap(subBitmap)
                            .setHasClip(command.needClip())
                            .setSnapShotCommand(command);
                    subFilter.release();
                    mainExport.addSubPhoto(subPhoto);

                    if (CameraInit.getConfig().isDebuggable()) {
                        Log.i(TAG, String.format("snapshot sub image (%d, %d) use times %d ", outWidth, outHeight, (SystemClock.uptimeMillis() - startTimes)));
                    }
                }
            }

            try {
                if (snapShotCommand.getCallback() != null) {
                    snapShotCommand.getCallback().onReceiveValue(mainExport);
                }
            } catch (Exception e) {
                CameraShould.fail("", e);
            }
        }


        if (mScreenRender != null) {
            mScreenRender.setInputSize(data.textureWidth, data.textureHeight);
            mScreenRender.setInputTextureId(data.texture);
            mScreenRender.setViewPort(mWindowWidth, mWindowHeight);
            mScreenRender.draw();
        }

        if (CameraInit.getConfig().isDebuggable()) {
            step4 = SystemClock.elapsedRealtimeNanos() - step4;

            if (SystemClock.uptimeMillis() - mResetRecordFpsTime > 1000) {
                Log.i(TAG, "camera video view fps:" + mFpsCount);
                mFpsCount = 0;
                mResetRecordFpsTime = SystemClock.uptimeMillis();
            } else {
                mFpsCount++;
            }

            Log.i(TAG, String.format(Locale.CHINA, "draw use time : %.3f (1: %.3f , 2:%.3f, 3:%.3f , 4:%.3f) ",
                    (float) (SystemClock.elapsedRealtimeNanos() - start) / 1000000, (float) step1 / 1000000, (float) step2 / 1000000, (float) step3 / 1000000, (float) step4 / 1000000)
            );
        }

    }

    private void calculateCameraShowLayout() {
        if (mCameraSurfaceRender == null) {
            return;
        }
        mTempRectF.setEmpty();

        switch (mCameraScaleType) {
            case ViewPort.CENTER_INSIDE:
                //CameraShould.fail("not support now");
                break;
            case ViewPort.FILL_CENTER:
                mTempRectF.set(mMarginPercentage[0] * mWindowWidth,
                        mMarginPercentage[1] * mWindowHeight,
                        (1 - mMarginPercentage[2]) * mWindowWidth,
                        (1 - mMarginPercentage[3]) * mWindowHeight
                );
                break;
            default:
                CameraShould.fail("not support now");
                break;
        }

        if (mLayoutChangeListener != null && !mCameraShowRect.equals(mTempRectF)) {
            mCameraShowRect.set(mTempRectF);
            mLayoutChangeListener.onCameraShowLayoutChanged(mTempRectF.left, mTempRectF.top, mTempRectF.width(),
                    mTempRectF.height());
            mTempRectF.setEmpty();
        }
    }

    public RectF getCameraShowRect() {
        return mCameraShowRect;
    }


    private ICameraRender mCameraSurfaceRender;


    public void setCameraRender(ICameraRender cameraRender) {
        mCameraSurfaceRender = cameraRender;
        if (mCameraSurfaceRender != null) {
            mCameraSurfaceRender.setCameraFrameAvailableListener(this);
        }
    }


    private void initSurfaceTexture() {
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        mSurfaceTextureId = textures[0];

        GLES20.glDisable(GLES20.GL_DEPTH_TEST);
        GLES20.glDisable(GLES20.GL_CULL_FACE);
        GLES20.glDisable(GLES20.GL_BLEND);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, mSurfaceTextureId);
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

    }


    public void setCameraScaleType(int cameraScaleType) {
        if (mCameraScaleType != cameraScaleType) {
            mLayoutChange = true;
        }
        mCameraScaleType = cameraScaleType;
        requestRenderIfNeed();
    }

    public int getCameraScaleType() {
        return mCameraScaleType;
    }

    @Override
    public void onCameraFrameAvailable() {
        mHost.requestRender();
    }
}
