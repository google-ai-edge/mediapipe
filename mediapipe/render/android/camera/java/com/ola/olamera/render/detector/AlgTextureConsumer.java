package com.ola.olamera.render.detector;

/*
 *
 *  Creation    :  2021/1/25
 *  Author      : jiaming.wjm@
 */

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.PixelFormat;
import android.media.Image;
import android.media.ImageReader;
import android.opengl.EGLContext;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.Surface;

import androidx.annotation.RequiresApi;

import com.ola.olamera.camera.anotaion.ExecutedBy;
import com.ola.olamera.util.CameraLogger;
import com.ola.olamera.util.CameraShould;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.ola.olamera.render.detector.AlgTextureConsumer.OutputType.NV21;


@RequiresApi(api = Build.VERSION_CODES.KITKAT)
public class AlgTextureConsumer implements ImageReader.OnImageAvailableListener, IAlgTextureConsumer {


    private NV21Buffer mNV21Buffer;
    private Handler mHandler;
    private ImageReader mImageReader;

    private GLAlgSurfaceProcessPipe mGLAlgSurfaceProcessPipe;

    private OnAlgCpuDataReceiver mOnAlgCpuDataReceiver;

    private Surface mSurface;

    private final Context mContext;

    public enum OutputType {
        RGBA_8888,
        NV21
    }

    private OutputType mOutputType = NV21;


    public AlgTextureConsumer(Context context) {
        mContext = context;
    }


    private void startThread() {
        if (mHandler == null) {
            HandlerThread handlerThread = new HandlerThread("elgCodec");
            handlerThread.start();
            mHandler = new Handler(handlerThread.getLooper());
        }
    }

    private int mInWidth;
    private int mInHeight;


    public void updateInputTexture(EGLContext eglContext, int textureId, int width, int height, long timeStamp) {
        if (isReleased()) {
            return;
        }
        startThread();

        synchronized (AlgTextureConsumer.this) {
            if (mGLDrawCommandList.size() > 2) {
                mGLDrawCommandList.remove(0);
            }
            mGLDrawCommandList.add(new GLDrawCommand(eglContext, textureId, width, height, timeStamp));
        }

        doLatestCommand();
    }

    private synchronized void doLatestCommand() {
        if (isReleased()) {
            return;
        }
        if (mProcessingCommand == null) {

            //获取最新帧
            int nextIndex = mGLDrawCommandList.size() - 1;
            if (nextIndex >= 0) {
                mProcessingCommand = mGLDrawCommandList.remove(nextIndex);
                mGLDrawCommandList.clear();
            }

            if (mProcessingCommand != null) {
                mHandler.post(mProcessingCommand);
            }
        }
    }


    /**
     * 当前只缓存最后的那一帧
     */
    private final List<GLDrawCommand> mGLDrawCommandList = new ArrayList<>(1);

    private GLDrawCommand mProcessingCommand;

    private class GLDrawCommand implements Runnable {
        EGLContext eglContext;
        int textureId;
        int width;
        int height;
        long timeStamp;

        public GLDrawCommand(EGLContext eglContext, int textureId, int width, int height, long timeStamp) {
            this.eglContext = eglContext;
            this.textureId = textureId;
            this.width = width;
            this.height = height;
            this.timeStamp = timeStamp;
        }

        @Override
        public void run() {
            if (isReleased()) {
                return;
            }
            try {
                onInputTextureSizeChange(eglContext, width, height);
                mGLAlgSurfaceProcessPipe.draw(textureId, timeStamp);
            } catch (Exception e) {
                mProcessingCommand = null;
                doLatestCommand();
            }
        }
    }

    private boolean isReleased() {
        return mState == STATE.RELEASED;
    }

    @ExecutedBy("GL Thread")
    private void onInputTextureSizeChange(EGLContext eglContext, int inWidth, int inHeight) {
        if (mInWidth == inWidth && mInHeight == inHeight) {
            return;
        }
        if (inWidth == 0 || inHeight == 0) {
            return;
        }
        initGLContext(eglContext, inWidth, inHeight);
        mInWidth = inWidth;
        mInHeight = inHeight;
    }

    @SuppressLint("WrongConstant")
    @ExecutedBy("GL Thread")
    private void initGLContext(EGLContext eglContext, int inWidth, int inHeight) {
        int surfaceWidth;
        int surfaceHeight;
        if (inWidth > 720) {
            surfaceWidth = 720;
            surfaceHeight = (int) (((float) inHeight) * 720 / inWidth);
        } else {
            surfaceWidth = inWidth;
            surfaceHeight = inHeight;
        }

        //因为要转化成nv21数据格式，所以要求宽高是2的倍数，这里用4的倍数，是因为二维码库需要，差几个相素影响并不大，所以还是用4的倍数
        surfaceWidth = surfaceWidth - surfaceWidth % 4;
        surfaceHeight = surfaceHeight - surfaceHeight % 4;

        releaseSurface();

        mImageReader = ImageReader.newInstance(surfaceWidth, surfaceHeight, PixelFormat.RGBA_8888, 1
                /**
                 * maxImages = 1 , use {@link ImageReader#acquireNextImage()} to obtain latest image
                 *
                 * @see ImageReader#acquireNextImage()
                 * @see ImageReader#acquireLatestImage()
                 */
        );
        mImageReader.setOnImageAvailableListener(this, mHandler);

        mSurface = mImageReader.getSurface();

        mGLAlgSurfaceProcessPipe = new GLAlgSurfaceProcessPipe(mContext, surfaceWidth, surfaceHeight, mSurface, eglContext);

        mGLAlgSurfaceProcessPipe.prepareFilter(inWidth, inHeight);
    }

    @ExecutedBy("GL Thread")
    private void releaseSurface() {
        if (mGLAlgSurfaceProcessPipe != null) {
            mGLAlgSurfaceProcessPipe.release();
            mGLAlgSurfaceProcessPipe = null;
        }

        /**
         * 因为ImageReader的{@link android.media.ImageReader.OnImageAvailableListener}和 close在相同的线程。所以不会存在
         * 关闭后还尝试获取{@link ImageReader#acquireLatestImage()} }的行为风险
         */
        if (mImageReader != null) {
            mImageReader.close();
            mImageReader = null;
        }

        if (mSurface != null) {
            mSurface.release();
            mSurface = null;
        }
    }

    public static long lastTimeStamp = Long.MIN_VALUE;

    @Override
    @ExecutedBy("GL Thread")
    public void onImageAvailable(ImageReader reader) {
        Image image = null;
        try {
            if (isReleased()) {
                return;
            }

            /**
             *
             * {@link ImageReader#acquireLatestImage()} will case Warning (not fatal error）
             * Unable to acquire a lockedBuffer, very likely client tries to lock more than maxImages buffers
             *
             * https://stackoverflow.com/questions/36419722/error-unable-to-acquire-a-lockedbuffer-very-likely-client-tries-to-lock-more-t/46910375
             *
             * You may provide parameter maxImages as 1 for method ImageReader.newInstance. acquireLatestImage calls acquireNextSurfaceImage before close the only one image buffer, which lead to this warning.
             *
             * Use acquireNextImage in this case. If maxImages is bigger than 1, acquireLatestImage don't have this problem.
             */
            image = reader.acquireLatestImage();

            final int imageWidth = image.getWidth();
            final int imageHeight = image.getHeight();
            final Image.Plane[] planes = image.getPlanes();
            if (planes == null || planes.length == 0) {
                return;
            }
            Image.Plane plane = planes[0];

            if (plane == null) {
                return;
            }

            final ByteBuffer buffer = plane.getBuffer();
            int rowStride = plane.getRowStride();

            if (mOutputType == NV21) {
                int yuv_row = (int) (((float) imageHeight) * (1f / 4/*y height*/ + 1f / 8/*uv height*/));
                int yuv_row_stride = imageWidth * 4;

                if (mNV21Buffer == null) {
                    mNV21Buffer = new NV21Buffer(imageWidth, imageHeight);
                } else {
                    //size change
                    NV21Buffer temp = mNV21Buffer;
                    temp.readLock();
                    if (temp.getWidth() != imageWidth || temp.getHeight() != imageHeight) {
                        mNV21Buffer = new NV21Buffer(imageWidth, imageHeight);
                    }
                    temp.readUnlock();
                }

                //java层如果频繁创建byte会导致GC繁忙，这里通过读写锁的方式保证数据一致性
                byte[] yuvData = mNV21Buffer.writeLock();
                try {
                    for (int i = 0; i < yuv_row; i++) {
                        buffer.position(i * rowStride);
                        buffer.get(yuvData, i * yuv_row_stride, yuv_row_stride);
                    }
                } finally {
                    mNV21Buffer.writeUnlock();
                }


                if (mOnAlgCpuDataReceiver != null) {
                    if (lastTimeStamp >= image.getTimestamp()) {
                        CameraShould.fail("wujm ----- fail " + lastTimeStamp + " > " + image.getTimestamp());
                    }
                    lastTimeStamp = image.getTimestamp();
                    mOnAlgCpuDataReceiver.onReceiveCpuData(mNV21Buffer, image.getTimestamp());
                }
            }
        } catch (Exception e) {
            CameraShould.fail("", e);
        } finally {
            if (image != null) {
                image.close();
            }

            synchronized (AlgTextureConsumer.this) {
                mProcessingCommand = null;
            }
            doLatestCommand();
        }
    }


    private enum STATE {
        RELEASED,
        INIT
    }

    private STATE mState = STATE.INIT;


    public void release() {
        if (isReleased()) {
            return;
        }
        synchronized (this) {
            mState = STATE.RELEASED;
            mProcessingCommand = null;
            mGLDrawCommandList.clear();
        }

        if (mHandler == null) {
            return;
        }

        final Handler handler = mHandler;
        mHandler = null;

        handler.post(() -> releaseInner(handler));

    }

    @Override
    public void resume() {
        //TODO
    }

    @Override
    public void pause() {
        //TODO
    }

    @Override
    public void setOnAlgCpuDataReceiver(OnAlgCpuDataReceiver receiver) {
        mOnAlgCpuDataReceiver = receiver;
    }

    @ExecutedBy("GL Thread")
    private void releaseInner(Handler handler) {

        releaseSurface();

        if (handler != null) {
            handler.getLooper().quitSafely();
        }
    }

}
