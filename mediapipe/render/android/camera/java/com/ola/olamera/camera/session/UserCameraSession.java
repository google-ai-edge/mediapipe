package com.ola.olamera.camera.session;
/*
 *
 *  Creation    :  20-11-23
 *  Author      : jiaming.wjm@
 */

import android.os.Build;

import com.ola.olamera.camera.camera.Camera2Info;
import com.ola.olamera.camera.camera.CameraLifeManager;
import com.ola.olamera.camera.camera.CameraState;
import com.ola.olamera.camera.session.config.CameraSelectConfig;

import java.util.concurrent.Executor;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.lifecycle.MutableLiveData;


/**
 * 不直接通过open,close的行为来直接使用相机,而是通过抽象窗口Session,使用Session的生命事件来控制相机
 * <p>
 * {@link IUserCameraSession#active()} 开启相机
 * 　{@link IUserCameraSession#inactive()} 关闭相机
 */
@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public final class UserCameraSession implements IUserCameraSession {

    private State mState = State.INACTIVE;

    private CameraSelector mCameraSelector;

    private final CameraLifeManager mCameraLifeManager;

    private final SessionConfig mSessionConfig;


    public UserCameraSession(@NonNull CameraLifeManager cameraLifeManager, @NonNull SessionConfig sessionConfig) {
        mCameraLifeManager = cameraLifeManager;
        mSessionConfig = sessionConfig;
    }

    public UserCameraSession setCameraSelector(CameraSelector cameraSelector) {
        mCameraSelector = cameraSelector;
        return this;
    }


    private boolean openCamera() {
        if (mCameraSelector == null) {
            return false;
        }
        mSessionConfig.setSelectConfig(new CameraSelectConfig(mCameraSelector));
        return mCameraLifeManager.openCamera(mCameraSelector, mSessionConfig);
    }

    public Camera2Info getCamera2Info() {
       return mCameraLifeManager.getCamera2Info();
    }

    public void enableFlash(boolean enable, CameraCaptureCallback callback, Executor executor) {
        mCameraLifeManager.enableFlash(enable, callback, executor);
    }

    public void takePicture(@NonNull SingleCaptureConfig singleCaptureConfig,
                            @NonNull ImageCapture.OnImageCapturedCallback capturedCallback) {
        mCameraLifeManager.takePicture(singleCaptureConfig, capturedCallback);
    }

    public void closeCamera() {
        mCameraLifeManager.closeCamera();
    }

    public @Nullable
    MutableLiveData<CameraState> getCameraState() {
        return mCameraLifeManager.getCurrentCameraState();
    }


    @Override
    public boolean active() {
        if (mState == State.ACTIVE) {
            return false;
        }
        mState = State.ACTIVE;
        return openCamera();
    }


    @Override
    public boolean inactive() {
        if (mState == State.INACTIVE) {
            return false;
        }
        mState = State.INACTIVE;
        closeCamera();
        return true;
    }

    @Override
    public boolean isActive() {
        return mState == State.ACTIVE;
    }
}
