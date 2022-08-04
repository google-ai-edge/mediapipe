package com.ola.olamera.render.detector;
/*
 *
 *  Creation    :  2021/1/26
 *  Author      : jiaming.wjm@
 */

import android.content.Context;
import android.opengl.GLES20;
import android.util.Log;


import com.ola.olamera.render.IGLFilter;
import com.ola.olamera.util.GlCommonUtil;
import com.ola.olamera.util.MatrixUtils;
import com.ola.olamera.util.OpenGlUtils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static android.opengl.GLES10.GL_NO_ERROR;

public class Rgba2YuvFilter implements IGLFilter {


    public static String RGBA_2_YUV_FRAGMENT = null;
    public static String RGBA_2_YUV_VERTEX = null;


    private final FloatBuffer mVtxBuf = OpenGlUtils.createSquareVtx();
    private final float[] mIdentityMtx = OpenGlUtils.createIdentityMtx();

    protected int mInputTextureId = -1;

    private final int[] mTexId = new int[]{0};

    private int mFboId = -1;
//    private int mRboId = -1;

    protected int mInputWidth = -1;
    protected int mInputHeight = -1;

    protected int mOutputWidth = -1;
    protected int mOutputHeight = -1;

    private boolean mPrepared = false;

    private final Context mContext;

    public Rgba2YuvFilter(Context context) {
        mContext = context;
    }


    public void setInputSize(int width, int height) {
        mInputWidth = width;
        mInputHeight = height;
        mOutputWidth = mInputWidth;
        mOutputHeight = mInputHeight;
    }

    public int getOutputWidth() {
        return mOutputWidth;
    }

    public int getOutputHeight() {
        return mOutputHeight;
    }

    public void clear() {
        GLES20.glClearColor(0f, 0f, 0f, 1f);
        GLES20.glClear(GLES20.GL_DEPTH_BUFFER_BIT | GLES20.GL_COLOR_BUFFER_BIT);
    }

    public void prepare() {
        Log.d("e", "prepare createFrameBuffer" + mPrepared + ", width " + mInputWidth + ", height" + mInputHeight);
        if (mPrepared) {
            return;
        }

        loadShaderAndParams();
        createEffectTexture();
        mPrepared = true;
    }

    public void setInputTextureId(int textureId) {
        mInputTextureId = textureId;
    }


    private int mMVPHandler = -1;
    private int mInputImageTextureWidthHandler = -1;
    private int mInputImageTextureHeightHandler = -1;
    private int mProgram = -1;
    private int mPositionHandle = -1;
    private int mInputTexCoordHandle = -1;


    private void loadShaderAndParams() {

        if (RGBA_2_YUV_VERTEX == null) {
            RGBA_2_YUV_VERTEX = OpenGlUtils.readRawShaderFile(mContext, "rgba_2_yuv_vertex.vert");
        }

        if (RGBA_2_YUV_FRAGMENT == null) {
            RGBA_2_YUV_FRAGMENT = OpenGlUtils.readRawShaderFile(mContext, "rbga_2_yuv_frag.frag");
        }

        checkGlError("initSH_S");

        mProgram = GlCommonUtil.createProgram(RGBA_2_YUV_VERTEX, RGBA_2_YUV_FRAGMENT);

        if (mProgram == 0) {
            //TODO 错误码
        }

        mInputTexCoordHandle = GLES20.glGetAttribLocation(mProgram, "inputTextureCoordinate");
        mPositionHandle = GLES20.glGetAttribLocation(mProgram, "position");

        mMVPHandler = GLES20.glGetUniformLocation(mProgram, "mvp");
        mInputImageTextureWidthHandler = GLES20.glGetUniformLocation(mProgram, "inputImageTextureWidth");
        mInputImageTextureHeightHandler = GLES20.glGetUniformLocation(mProgram, "inputImageTextureHeight");

        checkGlError("initSH_E");
    }

    public int getInputWidth() {
        return mInputWidth;
    }

    public int getInputHeight() {
        return mInputHeight;
    }

    private void createEffectTexture() {
        if (mInputWidth <= 0 || mInputHeight <= 0) {
            return;
        }
        checkGlError("initFBO_S");
        createFrameBuffer();
        GLES20.glGenTextures(1, mTexId, 0);

//        GLES20.glBindRenderbuffer(GLES20.GL_RENDERBUFFER, mRboId);
//        GLES20.glRenderbufferStorage(GLES20.GL_RENDERBUFFER,
//                GLES20.GL_DEPTH_COMPONENT16, mOutputWidth, mOutputHeight);

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, mFboId);
//        GLES20.glFramebufferRenderbuffer(GLES20.GL_FRAMEBUFFER,
//                GLES20.GL_DEPTH_ATTACHMENT, GLES20.GL_RENDERBUFFER, mRboId);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mTexId[0]);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA,
                mOutputWidth, mOutputHeight, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);

        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER,
                GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, mTexId[0], 0);

        if (GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER) !=
                GLES20.GL_FRAMEBUFFER_COMPLETE) {
            throw new RuntimeException("glCheckFramebufferStatus()");
        }
        checkGlError("initFBO_E");
    }


    public int getOutputTextureId() {
        return mTexId[0];
    }


    public void draw() {
        if (-1 == mProgram
                || mInputTextureId == -1
                || mInputWidth == -1
                || mInputHeight == -1) {
            return;
        }

        checkGlError("draw_S");
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, mFboId);

        checkGlError("draw_S");

        GLES20.glViewport(0, 0, mOutputWidth, mOutputHeight);

        checkGlError("draw_S");

        GLES20.glClearColor(0f, 1f, 0f, 1f);
        GLES20.glClear(GLES20.GL_DEPTH_BUFFER_BIT | GLES20.GL_COLOR_BUFFER_BIT);

        GLES20.glUseProgram(mProgram);

        mVtxBuf.position(0);
        GLES20.glVertexAttribPointer(mPositionHandle, 3, GLES20.GL_FLOAT, false, 4 * (3 + 2), mVtxBuf);
        GLES20.glEnableVertexAttribArray(mPositionHandle);

        checkGlError("draw_S");

        mVtxBuf.position(3);
        GLES20.glVertexAttribPointer(mInputTexCoordHandle, 2, GLES20.GL_FLOAT, false, 4 * (3 + 2), mVtxBuf);
        GLES20.glEnableVertexAttribArray(mInputTexCoordHandle);

        checkGlError("draw_S");

        GLES20.glUniform1f(mInputImageTextureHeightHandler, mInputHeight);
        GLES20.glUniform1f(mInputImageTextureWidthHandler, mInputWidth);


        GLES20.glUniformMatrix4fv(mMVPHandler, 1, false, mIdentityMtx, 0);

        checkGlError("draw_S");

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mInputTextureId);

        checkGlError("draw_S");

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        checkGlError("draw_S");

        GLES20.glDisableVertexAttribArray(mPositionHandle);
        GLES20.glDisableVertexAttribArray(mInputTexCoordHandle);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glUseProgram(0);

        checkGlError("draw_E");
    }

    public static void checkGlError(String op) {
        int errorCode = GLES20.glGetError();
        if (errorCode != GL_NO_ERROR) {
            throw new IllegalStateException(String.format(Locale.CHINA, "GL Error when %s with code %d ", op, errorCode));
        }
    }


    private void createFrameBuffer() {
        int[] fboId = new int[]{0};
        int[] rboId = new int[]{0};
        GLES20.glGenFramebuffers(1, fboId, 0);
        GLES20.glGenRenderbuffers(1, rboId, 0);
        mFboId = fboId[0];
//        mRboId = rboId[0];
        Log.d("e", "createFrameBuffer: ");
    }

    private void releaseFrameBuffer() {
        if (mFboId != -1) {
            int[] fboId = new int[]{mFboId};
            GLES20.glDeleteFramebuffers(1, fboId, 0);
            mFboId = -1;
        }
//        if (mRboId != -1) {
//            int[] rboId = new int[]{mRboId};
//            GLES20.glDeleteRenderbuffers(1, rboId, 0);
//            mRboId = -1;
//        }
    }

    private void releaseProgram() {
        if (mProgram == -1) {
            return;
        }
        GLES20.glDeleteProgram(mProgram);
    }

    public void release() {
        releaseTexture();
        releaseFrameBuffer();
        releaseProgram();
    }

    public void releaseTexture(){
        if (mTexId[0] > 0) {
            GlCommonUtil.deleteGLTexture(mTexId);
        }
    }


}
