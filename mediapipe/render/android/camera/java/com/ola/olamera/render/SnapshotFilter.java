package com.ola.olamera.render;

import android.annotation.TargetApi;
import android.graphics.Bitmap;
import android.graphics.PointF;
import android.graphics.Rect;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.util.Log;

import com.ola.olamera.render.photo.ExportPhoto;
import com.ola.olamera.render.photo.SnapShotCommand;
import com.ola.olamera.util.CameraLogger;
import com.ola.olamera.util.GlCommonUtil;
import com.ola.olamera.util.MatrixUtils;
import com.ola.olamera.util.OpenGlUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.LinkedList;

import static com.ola.olamera.render.ScreenRenderFilter.getVexPositionMatrix;


@TargetApi(18)
//TODO 后面将filter相同内容抽象成基类
public class SnapshotFilter {

    private final FloatBuffer mVtxBuf = OpenGlUtils.createSquareVtx();
    private final float[] mPosMtx = MatrixUtils.flipF(OpenGlUtils.createIdentityMtx(), false, true);
    private final float[] mTexMtx = OpenGlUtils.createIdentityMtx();

    protected int mInputTextureId = -1;
    private int mProgram = -1;
    private int maPositionHandle = -1;
    private int maTexCoordHandle = -1;
    private int muPosMtxHandle = -1;
    private int muTexMtxHandle = -1;

    private final int[] mTexId = new int[]{0};

    private int mFboId = -1;

    protected int mInputWidth = -1;
    protected int mInputHeight = -1;

    protected int mOutputWidth = -1;
    protected int mOutputHeight = -1;

    private boolean mIsExternalOES;//是否从外部纹理读取数据

    private final LinkedList<Runnable> mRunOnDraw;
    private final String mVertex;
    private final String mFragment;

    private boolean mNeedClear = true;

    private boolean mPrepared = false;


    public SnapshotFilter() {
        mRunOnDraw = new LinkedList<>();
        mVertex = GLConstant.SHADER_DEFAULT_VERTEX;
        mFragment = GLConstant.SHADER_DEFAULT_FRAGMENT_NOT_OES;
    }


    public void setInputSize(int width, int height) {
        mInputWidth = width;
        mInputHeight = height;
        mOutputWidth = mInputWidth;
        mOutputHeight = mInputHeight;
    }


    public void setOutputSize(int width, int height) {
        mOutputWidth = width;
        mOutputHeight = height;
    }

    public int getOutputWidth() {
        return mOutputWidth;
    }

    public int getOutputHeight() {
        return mOutputHeight;
    }

    public void setNeedClear(boolean need) {
        mNeedClear = need;
    }


    public void prepare() {
        if (mPrepared) {
            return;
        }
        loadShaderAndParams(mVertex, mFragment);
        createEffectTexture();
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

    private void createEffectTexture() {
        if (mInputWidth <= 0 || mInputHeight <= 0) {
            return;
        }
        GlCommonUtil.checkGlError("initFBO_S");
        createFrameBuffer();
        GLES20.glGenTextures(1, mTexId, 0);

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, mFboId);

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
        GlCommonUtil.checkGlError("initFBO_E");
    }


    public int getOutputTextureId() {
        return mTexId[0];
    }

    protected void runOnDraw(final Runnable runnable) {
        synchronized (mRunOnDraw) {
            mRunOnDraw.addLast(runnable);
        }
    }

    protected void runPendingOnDrawTasks() {
        while (!mRunOnDraw.isEmpty()) {
            mRunOnDraw.removeFirst().run();
        }
    }


    public Bitmap snapshot(SnapShotCommand snapShotCommand) {
        if (-1 == mProgram || mInputTextureId == -1 || mInputWidth == -1) {
            return null;
        }

        GlCommonUtil.checkGlError("draw_S");
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, mFboId);

        GlCommonUtil.checkGlError("draw_S");

        GLES20.glViewport(0, 0, mOutputWidth, mOutputHeight);

        GlCommonUtil.checkGlError("draw_S");

        if (mNeedClear) {
            GLES20.glClearColor(0f, 1f, 0f, 1f);
            GLES20.glClear(GLES20.GL_DEPTH_BUFFER_BIT | GLES20.GL_COLOR_BUFFER_BIT);
        }

        GLES20.glUseProgram(mProgram);

        GlCommonUtil.checkGlError("draw_S");

        runPendingOnDrawTasks();

        GlCommonUtil.checkGlError("draw_S");

        mVtxBuf.position(0);
        GLES20.glVertexAttribPointer(maPositionHandle,
                3, GLES20.GL_FLOAT, false, 4 * (3 + 2), mVtxBuf);
        GLES20.glEnableVertexAttribArray(maPositionHandle);

        GlCommonUtil.checkGlError("draw_S");

        mVtxBuf.position(3);
        GLES20.glVertexAttribPointer(maTexCoordHandle,
                2, GLES20.GL_FLOAT, false, 4 * (3 + 2), mVtxBuf);
        GLES20.glEnableVertexAttribArray(maTexCoordHandle);

        GlCommonUtil.checkGlError("draw_S");

        if (muPosMtxHandle >= 0) {
            GLES20.glUniformMatrix4fv(muPosMtxHandle, 1, false, mPosMtx, 0);
        }

        GlCommonUtil.checkGlError("draw_S");

        if (muTexMtxHandle >= 0) {
            GLES20.glUniformMatrix4fv(muTexMtxHandle, 1, false, mTexMtx, 0);
        }

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mInputTextureId);

        GlCommonUtil.checkGlError("draw_S");

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        Bitmap result = snapshotInner(snapShotCommand);

        GlCommonUtil.checkGlError("draw_S");

        GLES20.glDisableVertexAttribArray(maPositionHandle);
        GLES20.glDisableVertexAttribArray(maTexCoordHandle);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
        GLES20.glUseProgram(0);

        GlCommonUtil.checkGlError("draw_E");

        return result;
    }


    private void createFrameBuffer() {
        int[] fboId = new int[]{0};
        GLES20.glGenFramebuffers(1, fboId, 0);
        mFboId = fboId[0];
        Log.d("e", "createFrameBuffer: ");
    }

    private void releaseFrameBuffer() {
        if (mFboId != -1) {
            int[] fboId = new int[]{mFboId};
            GLES20.glDeleteFramebuffers(1, fboId, 0);
            mFboId = -1;
        }
    }

    private void releaseProgram() {
        if (mProgram == -1) {
            return;
        }
        GLES20.glDeleteProgram(mProgram);
    }

    public void release() {
        if (mTexId[0] > 0) {
            GlCommonUtil.deleteGLTexture(mTexId);
        }
        releaseFrameBuffer();
        releaseProgram();
    }

    private Bitmap snapshotInner(SnapShotCommand snapShotCommand) {
        int[] viewport = new int[4];
        GLES20.glGetIntegerv(GLES20.GL_VIEWPORT, viewport, 0);
        int x = viewport[0];
        int y = viewport[1];
        int width = viewport[2];
        int height = viewport[3];

        Rect rect = snapShotCommand.getGLSnapshotRect(width, height, false);

        final ByteBuffer rgbaData = ByteBuffer.allocateDirect(rect.width() * rect.height() * 4);
        rgbaData.order(ByteOrder.nativeOrder());

        GLES20.glReadPixels(rect.left, rect.top, rect.width(), rect.height(), GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, rgbaData);

        Bitmap bitmap = Bitmap.createBitmap(rect.width(), rect.height(), Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(rgbaData);

        CameraLogger.i("CameraVideoView", "take snapshot " + GLES20.glGetError());

        return bitmap;
    }

}
