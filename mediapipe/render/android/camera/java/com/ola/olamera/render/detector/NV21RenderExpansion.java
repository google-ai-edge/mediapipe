package com.ola.olamera.render.detector;
/*
 *
 *  Creation    :  2021/12/22
 *  Author      : jiaming.wjm@
 */

import static com.ola.olamera.render.detector.IAlgDetector.InputDataType.NV21;

import android.content.Context;
import android.opengl.EGL14;

import androidx.annotation.NonNull;

import com.ola.olamera.render.entry.RenderFlowData;
import com.ola.olamera.render.expansion.IRenderExpansion;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class NV21RenderExpansion implements IRenderExpansion {

    private IAlgTextureConsumer mNv21TextureConsumer;

    public final ConcurrentHashMap<Class<?>, IAlgDetector<?>> mDetectors = new ConcurrentHashMap<>();

    public final ConcurrentHashMap<Class<?>, OnDetectListener<?>> mDetectorListener = new ConcurrentHashMap<>();

    private Context mContext;

    public NV21RenderExpansion(Context context) {
        mContext = context;
    }

    public interface OnDetectListener<T> {
        void onDetect(T result);

    }

    public NV21RenderExpansion addDetector(@NonNull Class<?> clz,
                                           @NonNull IAlgDetector<?> detector) {
        mDetectors.put(clz, detector);
        return this;
    }


    private boolean isNeedNv21Stream() {
        for (Map.Entry<Class<?>, IAlgDetector<?>> entry : mDetectors.entrySet()) {
            IAlgDetector<?> detector = entry.getValue();
            if (detector.getState() == IAlgDetector.State.RUNNING
                    && detector.getInputDataType() == NV21) {
                return true;
            }
        }
        return false;
    }

    private AlgTextureConsumerCompat createAlgTextureConsumer(Context context) {
        AlgTextureConsumerCompat consumer = new AlgTextureConsumerCompat(context);
        consumer.setOnAlgCpuDataReceiver((buffer, timestamp) -> {
            for (Map.Entry<Class<?>, IAlgDetector<?>> entry : mDetectors.entrySet()) {
                Class<?> clz = entry.getKey();
                IAlgDetector<?> detector = entry.getValue();
                if (detector.getState() == IAlgDetector.State.RUNNING) {
                    OnDetectListener listener = mDetectorListener.get(clz);
                    if (listener != null) {
                        Object result = detector.detect(buffer, timestamp);
                        //类型转换肯定没有问题
                        listener.onDetect(result);
                    }
                }
            }
        });
        return consumer;
    }

    @NonNull
    @Override
    public RenderFlowData render(@NonNull RenderFlowData input, long timestamp) {
        if (isNeedNv21Stream()) {
            if (mNv21TextureConsumer == null) {
                mNv21TextureConsumer = createAlgTextureConsumer(mContext);
            }
            mNv21TextureConsumer.updateInputTexture(
                    EGL14.eglGetCurrentContext(), input.texture, input.textureWidth, input.textureHeight, timestamp);
        }
        return input;
    }


    public <Result, T extends IAlgDetector<Result>> void startDetector(@NonNull Class<T> clz, OnDetectListener<Result> detectListener) {
        IAlgDetector<?> detector = mDetectors.get(clz);
        if (detector == null) {
            throw new IllegalArgumentException("No support detect " + clz);
        }
        if (detector.getState() == IAlgDetector.State.UNINITIALIZED) {
            detector.init();
        }

        detector.start();

        if (detectListener != null) {
            mDetectorListener.put(clz, detectListener);
        }
    }

    public <T extends IAlgDetector<?>> void stopDetector(@NonNull Class<T> clz) {
        IAlgDetector<?> detector = mDetectors.get(clz);
        if (detector == null) {
            throw new IllegalArgumentException("No support detect " + clz);
        }
        detector.stop();
    }

    public void release() {
        for (Map.Entry<Class<?>, IAlgDetector<?>> entry : mDetectors.entrySet()) {
            IAlgDetector<?> detector = entry.getValue();
            detector.release();
        }
    }


    public <T extends IAlgDetector<?>> T getDetector(Class<T> clz) {
        IAlgDetector<?> detector = mDetectors.get(clz);
        if (detector == null) {
            throw new IllegalArgumentException("No support detect " + clz);
        }
        return (T) detector;
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        if (mNv21TextureConsumer == null) {
            mNv21TextureConsumer = createAlgTextureConsumer(mContext);
        }
    }

    @Override
    public void onSurfaceDestroy() {
        if (mNv21TextureConsumer != null) {
            mNv21TextureConsumer.release();
            mNv21TextureConsumer = null;
        }
    }
}
