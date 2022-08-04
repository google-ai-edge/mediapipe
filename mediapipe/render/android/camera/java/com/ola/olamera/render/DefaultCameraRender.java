package com.ola.olamera.render;
/*
 *
 *  Creation    :  2021/1/26
 *  Author      : jiaming.wjm@
 */

import android.opengl.EGLContext;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;

import com.ola.olamera.camera.camera.Camera2CameraImpl;
import com.ola.olamera.camera.camera.CameraState;
import com.ola.olamera.camera.camera.ICameraStateListener;
import com.ola.olamera.camera.preview.SurfaceTextureWrapper;
import com.ola.olamera.camera.session.CameraCaptureCallback;
import com.ola.olamera.camera.session.CameraCaptureComboCallback;
import com.ola.olamera.camera.session.CameraCaptureResult;
import com.ola.olamera.camerax.utils.CameraXCaptureResultManager;
import com.ola.olamera.util.Camera2CaptureResultManager;
import com.ola.olamera.util.CameraInit;
import com.ola.olamera.util.CameraShould;

import java.util.Map;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.camera.camera2.internal.Camera2CameraCaptureResult;


public class DefaultCameraRender implements ICameraRender, ICameraStateListener {

    public SurfaceTextureWrapper mSurfaceTextureWrapper;
    private final float[] mOESMatrix = new float[16];


    public static final String TAG = "CameraRender";

    private long mLastUpdateTime = -1;
    private long mLastFPSTime = 0;
    private long mFPSCount = 0;


    private volatile boolean mReceiveFirstFrame = false;
    private volatile boolean mHasNewFrame = false;

    private ICameraFrameAvailableListener mAvailableListener;

    private final Camera2CameraImpl mCamera;


    private final CameraCaptureCallback mCaptureCallback = new CameraCaptureCallback() {
        @Override
        public void onCaptureCompleted(@NonNull CameraCaptureResult cameraCaptureResult) {
            super.onCaptureCompleted(cameraCaptureResult);
            mMemManager.cacheCaptureResult(cameraCaptureResult);
        }
    };

    private final Object mResultLock = new Object();

    private CameraCaptureResult mCurrentCaptureResult;
    private Camera2CameraCaptureResult mCameraXCurrentCaptureResult;

    private final CameraCaptureComboCallback mComboCallback;

    private final Camera2CaptureResultManager mMemManager;
    private final CameraXCaptureResultManager mReleaseManager;

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public DefaultCameraRender(@NonNull SurfaceTextureWrapper surfaceTextureWrapper, Camera2CameraImpl camera,
                               CameraCaptureComboCallback captureCallback) {
        mSurfaceTextureWrapper = surfaceTextureWrapper;
        mCamera = camera;

        mComboCallback = captureCallback;
        if (mComboCallback != null) {
            mComboCallback.addCallback(mCaptureCallback);
        }
        mMemManager = new Camera2CaptureResultManager();
        mReleaseManager = new CameraXCaptureResultManager();

        if (mCamera != null) {
            mCamera.getCameraStateImmediatelyObservable().addListener(this);
        }

        surfaceTextureWrapper.getSurfaceTexture().setOnFrameAvailableListener(surfaceTexture -> {

            long surfaceTimeStamp = surfaceTextureWrapper.getSurfaceTexture().getTimestamp();

            if (CameraInit.getConfig().isDebuggable()) {
                Log.d(TAG, "CameraCaptureCallback.onCaptureCompleted setOnFrameAvailableListener -- " + surfaceTimeStamp);
            }

            mReceiveFirstFrame = true;
            synchronized (DefaultCameraRender.this) {
                mHasNewFrame = true;
            }

            CameraCaptureResult captureResult = mMemManager.consumeMatchCaptureResult(surfaceTimeStamp);
            // 对于CameraX的处理，内存释放
            Camera2CameraCaptureResult cameraXCaptureResult = mReleaseManager.consumeMatchCaptureResult(surfaceTimeStamp);

            synchronized (mResultLock) {
                /*
                 * 释放上一帧的数据，立即释放，而不是等待GC
                 */
                mMemManager.releaseCaptureResultNow(mCurrentCaptureResult);

                mCurrentCaptureResult = captureResult;

                mReleaseManager.releaseCaptureResultNow(mCameraXCurrentCaptureResult);
                mCameraXCurrentCaptureResult = cameraXCaptureResult;
            }

            if (mAvailableListener != null) {
                mAvailableListener.onCameraFrameAvailable();
            }
        });
    }


    @Override
    public boolean update(EGLContext context, int textureId, Map<String, Object> cameraInfo) {
        if (!mSurfaceTextureWrapper.isValid()) {
            return false;
        }
        boolean hasNewFrame;

        synchronized (DefaultCameraRender.this) {
            hasNewFrame = mHasNewFrame;
            mHasNewFrame = false;
        }

        if (CameraInit.getConfig().isDebuggable()) {
            long currentTime = SystemClock.uptimeMillis();
            long useTime = currentTime - mLastUpdateTime;

            if (currentTime - mLastFPSTime > 1000) {
                Log.i(TAG, "FPS:" + mFPSCount);
                mFPSCount = 1;
                mLastFPSTime = currentTime;
            } else {
                mFPSCount++;
            }
            mLastUpdateTime = SystemClock.uptimeMillis();
            Log.i(TAG, "frame:" + useTime);
        }


        boolean attachSuccess = mSurfaceTextureWrapper.attachToGLContext(context.hashCode(), textureId);
        try {
            if (attachSuccess) {
                try {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                        if (mCamera != null) {
                            cameraInfo.put("device_rotation",
                                    mCamera.getCamera2Info().getImageRotationHelper().getDeviceRotation());
                            cameraInfo.put("camera_sensor_rotation",
                                    mCamera.getCamera2Info().getImageRotationHelper().getCameraSensorOrientation());

                        }
                        synchronized (mResultLock) {
                            //                            if (mCurrentCaptureResult != null) {
                            //                                cameraInfo.put("af_state", mCurrentCaptureResult.getAfState());
                            //                                cameraInfo.put("af_mode", mCurrentCaptureResult.getAfMode());
                            //                            }
                        }
                    }
                } catch (Exception ignore) {
                }

                mSurfaceTextureWrapper.updateTexImage();
            }
        } catch (Exception e) {
            CameraShould.fail("Surface.updateTexImage Error", e);
        }

        return hasNewFrame;
    }


    @Override
    public boolean needRender() {
        return mReceiveFirstFrame && mSurfaceTextureWrapper.isValid();
    }

    @Override
    public void setCameraFrameAvailableListener(ICameraFrameAvailableListener listener) {
        mAvailableListener = listener;
    }

    @Override
    public int[] getCameraCaptureSize() {
        return mSurfaceTextureWrapper.getSize();
    }

    @Override
    public float[] getOESMatrix() {
        mSurfaceTextureWrapper.getTransformMatrix(mOESMatrix);
        return mOESMatrix;
    }

    @Override
    public boolean isMatrixInverseWidthHeight() {
        return true;
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    public void destroySurface() {

        if (CameraInit.getConfig().isDebuggable()) {
            Log.d(TAG, "destroySurface");
        }
        mReceiveFirstFrame = false;
        mHasNewFrame = false;
        if (mSurfaceTextureWrapper != null) {
            mSurfaceTextureWrapper.release();
        }
        if (mComboCallback != null) {
            mComboCallback.removeCallback(mCaptureCallback);
        }

        if (mCamera != null) {
            mCamera.getCameraStateImmediatelyObservable().removeListener(this);
        }

        synchronized (mResultLock) {
            mCurrentCaptureResult = null;
            mCameraXCurrentCaptureResult = null;
        }

        if (mComboCallback != null) {
            mComboCallback.removeCallback(mCaptureCallback);
        }

        mMemManager.clear();
        mReleaseManager.clear();
    }


    @Override
    public Camera2CameraImpl getCamera() {
        return mCamera;
    }

    @Override
    public void onCameraStateChanged(CameraState cameraState) {
        if (cameraState == CameraState.CLOSING || cameraState == CameraState.CLOSED) {
            if (CameraInit.getConfig().isDebuggable()) {
                Log.d(TAG, "camera state: " + cameraState + " clear invalid capture result ");
            }
            synchronized (mResultLock) {
                mCurrentCaptureResult = null;
            }


            mMemManager.clear();
        }
    }

    public void cacheCameraXCaptureResult(Camera2CameraCaptureResult captureResult) {
        mReleaseManager.cacheCaptureResult(captureResult);
    }
}
