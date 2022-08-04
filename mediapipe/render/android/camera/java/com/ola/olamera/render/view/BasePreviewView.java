package com.ola.olamera.render.view;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ValueAnimator;
import android.app.ActivityManager;
import android.content.Context;
import android.content.pm.ConfigurationInfo;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.RectF;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import android.util.Rational;
import android.util.Size;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;

import com.ola.olamera.camera.preview.IPreviewView;
import com.ola.olamera.camera.preview.ViewPort;
import com.ola.olamera.camerax.controller.OnGestureDetectorListener;
import com.ola.olamera.render.CameraVideoRenderExecutor;
import com.ola.olamera.render.CameraVideoRenderPipe;
import com.ola.olamera.render.detector.RenderExpansionManager;
import com.ola.olamera.render.expansion.IRenderExpansion;
import com.ola.olamera.render.photo.SnapShotCommand;
import com.ola.olamera.util.CameraLogger;
import com.ola.olamera.util.CameraShould;
import com.ola.olamera.util.Should;

import java.util.concurrent.Executor;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * Camera2和CameraX PreviewView的基类
 *
 * @author : yushan.lj@
 * @date : 2021/9/30
 */
public abstract class BasePreviewView extends FrameLayout implements IPreviewView {
    private static final String TAG = "BasePreviewView";

    private RenderExpansionManager mManager;
    private final CameraVideoRenderPipe mRender;
    private final AndroidGLSurfaceView mSurfaceView;
    private static int GL_VERSION = -1;
    private final Executor mExecutor;
    private final View mMask;

    public BasePreviewView(@NonNull Context context, @Nullable AttributeSet attrs) {
        this(context, false);
    }

    /**
     * @param ZOrderOverlay 解决低版本2个SurfaceView叠加显示问题
     */
    public BasePreviewView(@NonNull Context context, boolean ZOrderOverlay) {
        super(context, null);
        mSurfaceView = new AndroidGLSurfaceView(context);


        if (Build.VERSION.SDK_INT <= 25) {
            mSurfaceView.getHolder().setFormat(PixelFormat.OPAQUE);
            mSurfaceView.setZOrderMediaOverlay(ZOrderOverlay);
        }

        if (!mSurfaceView.getPreserveEGLContextOnPause()) {
            mSurfaceView.setPreserveEGLContextOnPause(true);
        }

        final int glVersion = getGLVersion(context);
        Log.e(TAG, "GLVersion = " + glVersion);
        mSurfaceView.setEGLContextClientVersion(glVersion);

        if (glVersion == 2) {
            CameraLogger.e(TAG, "not support gl 3.0");
        }

        mSurfaceView.setPreserveEGLContextOnPause(true);
        mSurfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0); // Alpha used for plane blending.
        RenderExecutor executor = new RenderExecutor();
        mManager = new RenderExpansionManager(context, executor);
        mRender = new CameraVideoRenderPipe(context, mSurfaceView, mManager);
        mSurfaceView.setEGLContextFactory(new AndroidGLSurfaceView.EGLContextFactory() {

            @Override
            public EGLContext createContext(EGL10 egl, EGLDisplay display, EGLConfig eglConfig) {
                int[] attributeList = {0x3098, glVersion, EGL10.EGL_NONE};
                EGLContext eglContext = egl.eglCreateContext(display, eglConfig, EGL10.EGL_NO_CONTEXT, attributeList);
                Log.e(TAG, "create gl context");
                return eglContext;
            }


            @Override
            public void destroyContext(EGL10 egl, EGLDisplay display, EGLContext context) {
                Log.e(TAG, "begin destroy gl context");
                mRender.onSurfaceDestroy();
                if (!egl.eglDestroyContext(display, context)) {
                    Log.e("DefaultContextFactory", "display : " + display + " context : " + context);
                }
                Log.e(TAG, "finish destroy gl context");
            }
        });
        mSurfaceView.setRenderer(mRender);
        mSurfaceView.setRenderMode(AndroidGLSurfaceView.RENDERMODE_WHEN_DIRTY);
        mExecutor = new CameraVideoRenderExecutor(mSurfaceView);
        addView(mSurfaceView);

        mMask = new View(context);
        addView(mMask, new LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
    }


    private class RenderExecutor implements Executor {

        @Override
        public void execute(Runnable command) {
            mRender.queueTask(() -> {
                try {
                    command.run();
                } catch (Throwable throwable) {
                    CameraShould.fail("", throwable);
                }
            });
        }
    }


    public RenderExpansionManager getExpansionManager() {
        return mManager;
    }

    public void pause() {
        CameraLogger.i(TAG, "do pause");
//        mSurfaceView.onPause();
    }

    public void resume() {
        CameraLogger.i(TAG, "do resume");
//        mSurfaceView.onResume();
    }

    @Override
    public void snapshot(final SnapShotCommand snapShotCommand) {
        CameraLogger.i(TAG, "VideoView.java = do mRender.snapshot");
        mRender.snapshot(snapShotCommand);

        mSurfaceView.requestRender();
    }


    public void setOnGestureDetectorListener(OnGestureDetectorListener listener) {

    }

    private ValueAnimator mTakePhotoAnimator;

    @Override
    public void doTakePhotoAnimation() {
        post(this::doTakePhotoAnimationInner);
    }

    private void doTakePhotoAnimationInner() {
        if (mTakePhotoAnimator != null) {
            mTakePhotoAnimator.cancel();
        }

        mTakePhotoAnimator = ValueAnimator.ofFloat(0, 1, 0);
        mTakePhotoAnimator.setDuration(200);
        mTakePhotoAnimator.addListener(new AnimatorListenerAdapter() {
            @Override
            public void onAnimationStart(Animator animation) {
                super.onAnimationStart(animation);
                mMask.setVisibility(VISIBLE);
                mMask.setBackgroundColor(Color.BLACK);
            }

            @Override
            public void onAnimationCancel(Animator animation) {
                super.onAnimationCancel(animation);
                mMask.setVisibility(INVISIBLE);
            }

            @Override
            public void onAnimationEnd(Animator animation) {
                super.onAnimationEnd(animation);
                mMask.setVisibility(INVISIBLE);
            }
        });
        mTakePhotoAnimator.addUpdateListener(animation -> mMask.setAlpha((Float) animation.getAnimatedValue()));
        mTakePhotoAnimator.start();
    }


    public void setScaleType(@ViewPort.ScaleType int scaleType) {
        mRender.setCameraScaleType(scaleType);
    }

    @Override
    public int getViewRotation() {
        return 0;
    }

    @Override
    public int getScaleType() {
        return mRender.getCameraScaleType();
    }

    @Override
    public Rational getAspectRatio() {
        return mRender.getCameraShowRational();
    }

    @Override
    public int getViewHeight() {
        return getMeasuredWidth();
    }

    @Override
    public int getViewWidth() {
        return getMeasuredHeight();
    }

    @Override
    public RectF getCameraShowRect() {
        return mRender.getCameraShowRect();
    }

    private Size mCameraSurfaceSize;

    @Override
    public void updateCameraSurfaceSize(Size size) {
        mCameraSurfaceSize = size;
    }

    @Override
    public Size getCameraSurfaceSize() {
        return mCameraSurfaceSize;
    }

    @NonNull
    public CameraVideoRenderPipe getRender() {
        return mRender;
    }

    public Executor getGLExecutor() {
        return mExecutor;
    }

    public static synchronized int getGLVersion(Context context) {
        if (GL_VERSION != -1) {
            return GL_VERSION;
        }
        ConfigurationInfo cfgInfo;
        try {
            ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
            cfgInfo = am.getDeviceConfigurationInfo();
        } catch (Exception e) {
            cfgInfo = null;
        }

        //默认使用2作为GL版本
        GL_VERSION = 2;
        if (cfgInfo != null && cfgInfo.reqGlEsVersion >= 0x30000) {
            GL_VERSION = 3;
        } else {
            Log.e(TAG, "not support gl version 3 (" + (cfgInfo != null ? cfgInfo.reqGlEsVersion : "unknown") + ")");
        }

        return GL_VERSION;
    }

}
