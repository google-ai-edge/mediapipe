package com.ola.olamera.util;
/*
 *
 *  Creation    :  2021/5/26
 *  Author      : jiaming.wjm@
 */

import android.hardware.camera2.CaptureResult;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.ola.olamera.camera.session.CameraCaptureResult;
import com.ola.olamera.render.DefaultCameraRender;

import java.lang.ref.WeakReference;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Queue;

/**
 * 立即释放CaptureResult中CameraMetadataNative持有的Native内存
 * <p>
 * CameraMetadataNative 默认情况下是执行GC的时候，会释放C++内存，但是会不及时，因为Native Heap上涨不会导致GC触发，但是会导致OOM
 * <p>
 * 目前的方式通过反射实现，存在一定的兼容性风险；对于而已。GC+内存检测兼容性好点，但是性能可能没有那么好
 */
public class Camera2CaptureResultManager {

    public final static String TAG = "Camera2MemManager";
    private int mMaxCache = 40;
    private boolean mEnable = true;

    public Camera2CaptureResultManager() {
        this(40);
    }

    public Camera2CaptureResultManager(int maxCache) {
        this.mMaxCache = maxCache;
        mEnable = CameraInit.getConfig().enableHighMemoryGC();
    }


    private final Queue<WeakReference<CameraCaptureResult>> mCacheCacheResult = new LinkedList<>();

    public void cacheCaptureResult(CameraCaptureResult captureResult) {
        if (!mEnable) {
            return;
        }

        if (captureResult == null) {
            return;
        }
        synchronized (mCacheCacheResult) {
            if (mCacheCacheResult.size() > mMaxCache) {
                mCacheCacheResult.remove();
            }
            if (CameraInit.getConfig().isDebuggable()) {
                Log.d(TAG,
                        "CameraCaptureCallback.onCaptureCompleted collect -- " + captureResult.getTimestamp());
            }
            mCacheCacheResult.add(new WeakReference<>(captureResult));
        }
    }

    /**
     * 消费并且删除 相对于 surfaceTimeStamp 过期的Result
     * 基于假设 SurfaceTexture的队列timestamp一定是递增的
     */
    public CameraCaptureResult consumeMatchCaptureResult(long surfaceTimeStamp) {
        if (!mEnable) {
            return null;
        }
        synchronized (mCacheCacheResult) {
            Iterator<WeakReference<CameraCaptureResult>> it = mCacheCacheResult.iterator();
            while (it.hasNext()) {
                CameraCaptureResult result = it.next().get();
                if (result == null) {
                    it.remove();
                    continue;
                }
                if (result.getTimestamp() < surfaceTimeStamp) {
                    //移除掉时间比较小的
                    it.remove();
                }
                if (result.getTimestamp() == surfaceTimeStamp) {
                    it.remove();
                    return result;
                }
            }
            return null;
        }
    }

    public void clear() {
        synchronized (mCacheCacheResult) {
            mCacheCacheResult.clear();
        }
    }


    private static Field sCameraMetaDataNativeField;
    private static Method sCameraMetaDataCloseMethod;
    private static Class sClass;

    /**
     * 立即释放CaptureResult中CameraMetadataNative持有的Native内存
     * <p>
     * CameraMetadataNative 默认情况下是执行GC的时候，会释放C++内存，但是会不及时，因为Native Heap上涨不会导致GC触发，但是会导致OOM
     */
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public boolean releaseCaptureResultNow(CameraCaptureResult captureResult) {
        if (!mEnable) {
            return false;
        }

        if (captureResult == null || captureResult.getCaptureResult() == null) {
            return false;
        }
        try {
            if (sCameraMetaDataNativeField == null) {
                sCameraMetaDataNativeField = CaptureResult.class.getDeclaredField("mResults");
                sCameraMetaDataNativeField.setAccessible(true);
            }

            if (sClass == null) {
                sClass = Class.forName("android.hardware.camera2.impl.CameraMetadataNative");
            }

            if (sCameraMetaDataCloseMethod == null) {
                sCameraMetaDataCloseMethod = CameraReflection.findMethod(sClass, "finalize");
                sCameraMetaDataCloseMethod.setAccessible(true);
            }

        } catch (NoSuchFieldException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }

        if (sCameraMetaDataCloseMethod == null || sCameraMetaDataNativeField == null) {
            return false;
        }

        try {
            long timeState = captureResult.getTimestamp();
            sCameraMetaDataCloseMethod.invoke(sCameraMetaDataNativeField.get(captureResult.getCaptureResult()));
            Log.d(DefaultCameraRender.TAG, "CaptureFrameHelper.release --- " + timeState);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }


}
