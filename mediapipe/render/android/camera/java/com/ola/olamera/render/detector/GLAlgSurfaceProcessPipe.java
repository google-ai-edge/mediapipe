package com.ola.olamera.render.detector;

import android.content.Context;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLExt;
import android.opengl.EGLSurface;
import android.view.Surface;

import com.ola.olamera.render.GlFboFilter;
import com.ola.olamera.render.ScreenRenderFilter;
import com.ola.olamera.render.view.BasePreviewView;


public class GLAlgSurfaceProcessPipe {

    protected EGLDisplay eglDisplay;
    protected EGLConfig mEglConfig;
    protected EGLContext mCurrentEglContext;
    protected final EGLSurface eglSurface;

    protected ScreenRenderFilter mScreenRenderFilter;
    protected Rgba2YuvFilter mRgba2YuvFilter;

    protected GlFboFilter mYFlipFilter;

    protected final Context mContext;

    protected final int mOutputSurfaceWidth;
    protected final int mOutputSurfaceHeight;


    public GLAlgSurfaceProcessPipe(Context context, int surfaceWidth, int surfaceHeight, Surface surface, EGLContext eglContext) {

        if (surfaceWidth % 4 != 0 || surfaceHeight % 4 != 0) {
            throw new RuntimeException("eglCreateWindowSurface 失败！");
        }

        mOutputSurfaceWidth = surfaceWidth;
        mOutputSurfaceHeight = surfaceHeight;

        mContext = context;
        createEGLContext(eglContext);

        int[] attrib_list = {
                EGL14.EGL_NONE
        };

        //创建EGLSurface
        eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, mEglConfig, surface, attrib_list, 0);

        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            throw new RuntimeException("eglCreateWindowSurface 失败！");
        }

        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, mCurrentEglContext)) {
            throw new RuntimeException("eglMakeCurrent 失败！");
        }
    }

    public void prepareFilter(int inputWidth, int inputHeight) {
        mYFlipFilter = new GlFboFilter(false);
        mYFlipFilter.setInputSize(inputWidth, inputHeight);
        mYFlipFilter.setOutputSize(mOutputSurfaceWidth, mOutputSurfaceHeight);
        mYFlipFilter.prepare();

        mRgba2YuvFilter = new Rgba2YuvFilter(mContext);
        mRgba2YuvFilter.setInputSize(mYFlipFilter.getOutputWidth(), mYFlipFilter.getOutputHeight());
        mRgba2YuvFilter.prepare();

        mScreenRenderFilter = new ScreenRenderFilter();
        mScreenRenderFilter.setViewPort(mRgba2YuvFilter.getOutputWidth(), mRgba2YuvFilter.getInputHeight());
        mScreenRenderFilter.setInputSize(mRgba2YuvFilter.getOutputWidth(), mRgba2YuvFilter.getOutputHeight());
        mScreenRenderFilter.prepare();
    }


    private void createEGLContext(EGLContext eglContext) {
        //创建虚拟屏幕
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);

        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            throw new RuntimeException("eglGetDisplay failed");
        }

        int[] versions = new int[2];
        //初始化elgdisplay
        if (!EGL14.eglInitialize(eglDisplay, versions, 0, versions, 1)) {
            throw new RuntimeException("eglInitialize failed");
        }

        int glVersion = BasePreviewView.getGLVersion(mContext);

        int[] attr_list = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, glVersion == 2 ? EGL14.EGL_OPENGL_ES2_BIT : EGLExt.EGL_OPENGL_ES3_BIT_KHR,
                EGL14.EGL_NONE
        };

        EGLConfig[] configs = new EGLConfig[1];

        int[] num_configs = new int[1];

        //配置eglDisplay 属性
        if (!EGL14.eglChooseConfig(eglDisplay, attr_list, 0,
                configs, 0, configs.length,
                num_configs, 0)) {
            throw new IllegalArgumentException("eglChooseConfig#2 failed");
        }

        mEglConfig = configs[0];

        int[] ctx_attrib_list = {
                EGL14.EGL_CONTEXT_CLIENT_VERSION, glVersion,
                EGL14.EGL_NONE
        };

        //创建EGL 上下文
        if (eglContext == null) {
            mCurrentEglContext = EGL14.eglCreateContext(eglDisplay, mEglConfig, EGL14.EGL_NO_CONTEXT, ctx_attrib_list, 0);
        } else {
            mCurrentEglContext = EGL14.eglCreateContext(eglDisplay, mEglConfig, eglContext, ctx_attrib_list, 0);
        }

        if (mCurrentEglContext == EGL14.EGL_NO_CONTEXT) {
            throw new RuntimeException("EGL Context Error.");
        }
    }


    public void draw(int textureId, long timestamp) {
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, mCurrentEglContext)) {
            throw new RuntimeException("eglMakeCurrent 失败！");
        }

        if (textureId < 0) {
            throw new RuntimeException("input texture error " + textureId);

        }

        mYFlipFilter.setInputTextureId(textureId);
        mYFlipFilter.flipPosMtxY();
        mYFlipFilter.draw(null);
        textureId = mYFlipFilter.getOutputTextureId();

        mRgba2YuvFilter.setInputTextureId(textureId);
        mRgba2YuvFilter.draw();
        textureId = mRgba2YuvFilter.getOutputTextureId();

        mScreenRenderFilter.setInputTextureId(textureId);
        mScreenRenderFilter.flipPosMtxY();
        mScreenRenderFilter.draw();

        EGLExt.eglPresentationTimeANDROID(eglDisplay, eglSurface, timestamp);
        //交换数据，输出到mediacodec InputSurface中
        EGL14.eglSwapBuffers(eglDisplay, eglSurface);

        Rgba2YuvFilter.checkGlError("drawError");

    }


    public void release() {
        EGL14.eglDestroySurface(eglDisplay, eglSurface);
        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, mCurrentEglContext);
        EGL14.eglDestroyContext(eglDisplay, mCurrentEglContext);
        EGL14.eglReleaseThread();
        EGL14.eglTerminate(eglDisplay);
    }
}
