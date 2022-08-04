package com.ola.olamera.render;

import android.annotation.TargetApi;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.Matrix;

import com.ola.olamera.util.GlCommonUtil;
import com.ola.olamera.util.MatrixUtils;
import com.ola.olamera.util.OpenGlUtils;

import java.nio.FloatBuffer;


@TargetApi(18)
//TODO 后面将filter相同内容抽象成基类
public class ScreenRenderFilter {

    private final FloatBuffer mVtxBuf = OpenGlUtils.createSquareVtx();
    private float[] mPosMtx = OpenGlUtils.createIdentityMtx();
    private final float[] mNormalPosMtx = OpenGlUtils.createIdentityMtx();

    private final float[] mFlipXPosMtx = MatrixUtils.flipF(OpenGlUtils.createIdentityMtx(), true, false);
    private final float[] mFlipYPosMtx = MatrixUtils.flipF(OpenGlUtils.createIdentityMtx(), false, true);

    protected int mInputTextureId = -1;
    private int mProgram = -1;
    private int maPositionHandle = -1;
    private int maTexCoordHandle = -1;
    private int muPosMtxHandle = -1;
    private int muTexMtxHandle = -1;

    private int mInputWidth = -1;
    private int mInputHeight = -1;

    private int mViewPortWidth = -1;
    private int mViewPortHeight = -1;

    private final String mVertex;
    private final String mFragment;
    private final boolean mIsExternalOES = false;

    private int mX = 0;
    private int mY = 0;

    private boolean mNeedClear = true;

    private boolean mPrepared = false;

    public ScreenRenderFilter() {
        mVertex = GLConstant.SHADER_DEFAULT_VERTEX;
        mFragment = GLConstant.SHADER_DEFAULT_FRAGMENT_NOT_OES;
    }


    public void normalPosMtx() {
        mPosMtx = mNormalPosMtx;
    }

    public void flipPosMtxX() {
        mPosMtx = mFlipXPosMtx;
    }

    public void flipPosMtxY() {
        mPosMtx = mFlipYPosMtx;
    }

    public void setInputSize(int width, int height) {
        boolean isChange = mInputWidth != width || mInputWidth != height;
        mInputWidth = width;
        mInputHeight = height;
        if (isChange) {
            setScaleType();
        }
    }


    public void setViewPort(int width, int height) {
        boolean isChange = mViewPortWidth != width || mViewPortHeight != height;
        mViewPortWidth = width;
        mViewPortHeight = height;
        if (isChange) {
            setScaleType();
        }
    }

    public int getViewPortWidth() {
        return mViewPortWidth;
    }

    public int getViewPortHeight() {
        return mViewPortHeight;
    }

    public void setNeedClear(boolean need) {
        mNeedClear = need;
    }

    public void clear() {
        GLES20.glClearColor(0f, 0f, 0f, 1f);
        GLES20.glClear(GLES20.GL_DEPTH_BUFFER_BIT | GLES20.GL_COLOR_BUFFER_BIT);
    }

    public void prepare() {
        if (mPrepared) {
            return;
        }

        loadShaderAndParams(mVertex, mFragment);

        mPrepared = true;
    }

    public void setInputTextureId(int textureId) {
        mInputTextureId = textureId;
    }

    private void loadShaderAndParams(String vertex, String fragment) {
        GlCommonUtil.checkGlError("initSH_S");
        mProgram = GlCommonUtil.createProgram(vertex, fragment);
        maPositionHandle = GLES20.glGetAttribLocation(mProgram, "position");
        maTexCoordHandle = GLES20.glGetAttribLocation(mProgram, "inputTextureCoordinate");
        muPosMtxHandle = GLES20.glGetUniformLocation(mProgram, "uPosMtx");
        muTexMtxHandle = GLES20.glGetUniformLocation(mProgram, "uTexMtx");
        GlCommonUtil.checkGlError("initSH_E");
    }

    public int getInputWidth() {
        return mInputWidth;
    }

    public int getInputHeight() {
        return mInputHeight;
    }


    public void setScaleType() {

        if (mInputWidth <= 0
                || mInputHeight <= 0
                || mViewPortHeight <= 0
                || mViewPortWidth <= 0) {
            return;
        }
        mPosMtx = OpenGlUtils.createIdentityMtx();

        getVexPositionMatrix(mInputWidth, mInputHeight, mViewPortWidth, mViewPortHeight, mPosMtx);

    }

    public static void getVexPositionMatrix(int iWidth, int iHeight, int oWidth, int oHeight, float[] posMatrix) {
        float input_radio = (float) iHeight / (float) iWidth;
        float out_radio = (float) oHeight / (float) oWidth;
        if (input_radio > out_radio) {
            Matrix.scaleM(posMatrix, 0, 1f, (float) input_radio / (float) out_radio, 1.0F);
        } else {
            Matrix.scaleM(posMatrix, 0, (float) out_radio / (float) input_radio, 1f, 1.0F);
        }
    }


    private float[] mTexMtx = OpenGlUtils.createIdentityMtx();

    public float[] getPosMtx() {
        return mPosMtx;
    }


    public void draw() {
        if (-1 == mProgram || mInputTextureId == -1 || mInputWidth == -1) {
            return;
        }

        GlCommonUtil.checkGlError("draw_S");
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);


        GLES20.glViewport(mX, mY, mViewPortWidth, mViewPortHeight);


        if (mNeedClear) {
            GLES20.glClearColor(0f, 0f, 0f, 1f);
            GLES20.glClear(GLES20.GL_DEPTH_BUFFER_BIT | GLES20.GL_COLOR_BUFFER_BIT);
        }

        GLES20.glUseProgram(mProgram);


        mVtxBuf.position(0);
        GLES20.glVertexAttribPointer(maPositionHandle,
                3, GLES20.GL_FLOAT, false, 4 * (3 + 2), mVtxBuf);
        GLES20.glEnableVertexAttribArray(maPositionHandle);


        mVtxBuf.position(3);
        GLES20.glVertexAttribPointer(maTexCoordHandle,
                2, GLES20.GL_FLOAT, false, 4 * (3 + 2), mVtxBuf);
        GLES20.glEnableVertexAttribArray(maTexCoordHandle);


        if (muPosMtxHandle >= 0) {
            GLES20.glUniformMatrix4fv(muPosMtxHandle, 1, false, mPosMtx, 0);
        }

        if (muTexMtxHandle >= 0) {
            GLES20.glUniformMatrix4fv(muTexMtxHandle, 1, false, mTexMtx, 0);
        }

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        if (mIsExternalOES) {
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, mInputTextureId);
        } else {
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mInputTextureId);
        }


        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);


        GLES20.glDisableVertexAttribArray(maPositionHandle);
        GLES20.glDisableVertexAttribArray(maTexCoordHandle);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);

        if (mIsExternalOES) {
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0);
        } else {
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        }
        GLES20.glUseProgram(0);

        GlCommonUtil.checkGlError("draw_E");
    }


    private void releaseProgram() {
        if (mProgram == -1) {
            return;
        }
        GLES20.glDeleteProgram(mProgram);
    }

    public void release() {
        releaseProgram();
    }
}
