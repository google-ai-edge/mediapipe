package com.quark.quamera.camera.camera;
/*
 * Copyright (C) 2005-2019 UCWeb Inc. All rights reserved.
 *  Description :
 *
 *  Creation    :  20-11-24
 *  Author      : jiaming.wjm@alibaba-inc.com
 */

import android.content.Context;
import android.os.Build;
import android.util.Size;

import com.quark.quamera.camera.preview.IPreviewView;
import com.quark.quamera.camera.sensor.DisplayOrientationDetector;
import com.quark.quamera.camera.sensor.ImageRotationHelper;
import com.quark.quamera.camera.session.CameraCaptureCallback;
import com.quark.quamera.camera.session.CameraSelector;
import com.quark.quamera.camera.session.ImageCapture;
import com.quark.quamera.camera.session.SessionConfig;
import com.quark.quamera.camera.session.SingleCaptureConfig;
import com.quark.quamera.util.CameraLogger;
import com.quark.quamera.util.Preconditions;

import java.util.List;
import java.util.concurrent.Executor;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.Observer;

//保证一次只有一个相机
//所有相机的生命管理
@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class CameraLifeManager {
    private Camera2CameraImpl mCurrentCamera;
    private ImageCapture mImageCapture;

    private final CameraRepository mCameraRepository;

    private final DisplayOrientationDetector mOrientationHelper;


    private final Observer<CameraState> mCameraStateObserver = new Observer<CameraState>() {
        @Override
        public void onChanged(CameraState cameraState) {
            if (cameraState == CameraState.OPEN) {
                mOrientationHelper.start();
            } else if (cameraState == CameraState.CLOSED) {
                mOrientationHelper.stop();
            }
        }
    };


    public CameraLifeManager(Context context, CameraRepository cameraRepository) {
        mCameraRepository = cameraRepository;
        mOrientationHelper = new DisplayOrientationDetector(context, Camera2CameraImpl.TAG);
    }


    public boolean openCamera(@NonNull CameraSelector cameraSelector, @NonNull SessionConfig config) {
        Preconditions.checkState(cameraSelector != null);
        Preconditions.checkState(config != null);

        List<Camera2CameraImpl> mathCameraList = mCameraRepository.filterCamera(cameraSelector);

        if (mathCameraList == null) {
            return false;
        }

        if (!mathCameraList.iterator().hasNext()) {
            CameraLogger.uploadError("NotMatchCamera", "not found " + cameraSelector.toLogString());
            return false;
        }

        Camera2CameraImpl camera2Camera = mathCameraList.iterator().next();

        if (camera2Camera == null) {
            return false;
        }

        if (mCurrentCamera != null) {
            mCurrentCamera.close();
            mCurrentCamera.getCameraStateObservable().removeObserver(mCameraStateObserver);
            mCurrentCamera = null;
        }

        mCurrentCamera = camera2Camera;
        mCurrentCamera.updateDisplayRotationDetector(mOrientationHelper);

        try {
            CameraSurfaceHelper.configPreviewSize(config.getPreviewConfig(), mCurrentCamera);

            IPreviewView previewView = config.getPreviewConfig().getPreviewView();
            previewView.updateCameraSurfaceSize(
                    new Size(config.getPreviewConfig().getActualWidth(),
                            config.getPreviewConfig().getActualHeight()));

            if (config.getPreviewConfig().getImageReaders() != null) {
                CameraSurfaceHelper.configImageReader(
                        config.getPreviewConfig().getImageReaders(), mCurrentCamera,
                        new ImageRotationHelper(mCurrentCamera.getCamera2Info(), mOrientationHelper)
                );
            }

            if (config.getImageCapture() != null) {
                CameraSurfaceHelper.configImageCaptureSize(config.getImageCapture(),
                        mCurrentCamera,
                        new ImageRotationHelper(mCurrentCamera.getCamera2Info(), mOrientationHelper));


                config.getImageCapture().bindCamera(mCurrentCamera);
                config.getImageCapture().bindPreviewView(config.getPreviewConfig().getPreviewView());
                mImageCapture = config.getImageCapture();
            }


        } catch (Exception e) {
            Preconditions.onException(e);
            return false;
        }

        mCurrentCamera.updateSessionConfig(config);
        mCurrentCamera.resetCaptureSession();
        mCurrentCamera.open();

        mCurrentCamera
                .getCameraStateObservable()
                .observeForever(mCameraStateObserver);
        return true;
    }


    public void enableFlash(boolean flash, CameraCaptureCallback callback, Executor executor) {
        if (mCurrentCamera != null) {
            mCurrentCamera.getControl().enableFlash(flash, callback, executor);
        }
    }

    public void takePicture(@NonNull SingleCaptureConfig singleCaptureConfig,
                            @NonNull ImageCapture.OnImageCapturedCallback capturedCallback) {
        if (mImageCapture != null) {
            mImageCapture.takePicture(singleCaptureConfig, capturedCallback);
        } else {
            capturedCallback.onError(new IllegalStateException("not config image capture"));
        }
    }

    public MutableLiveData<CameraState> getCurrentCameraState() {
        return mCurrentCamera != null ? mCurrentCamera.getCameraStateObservable() : null;
    }

    public Camera2Info getCamera2Info(){
        return mCurrentCamera != null ? mCurrentCamera.getCamera2Info() : null;
    }

    public void closeCamera() {
        if (mCurrentCamera != null) {
            mCurrentCamera.close();
            mOrientationHelper.forceQuit();
            mCurrentCamera.getCameraStateObservable().removeObserver(mCameraStateObserver);
            mCurrentCamera = null;
            mImageCapture = null;
        }
    }

}
