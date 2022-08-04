package com.ola.olamerademo;
/*
 *
 *  Creation    :  20-11-25
 *  Author      : jiaming.wjm@
 */

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.Surface;

import androidx.annotation.GuardedBy;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.Observer;


import com.ola.olamera.camera.Camera2Manager;
import com.ola.olamera.camera.camera.CameraState;
import com.ola.olamera.camera.concurrent.MainThreadExecutor;
import com.ola.olamera.camera.preview.IPreviewSurfaceProvider;
import com.ola.olamera.camera.preview.IPreviewView;
import com.ola.olamera.camera.preview.SurfaceTextureWrapper;
import com.ola.olamera.camera.session.CameraCaptureCallback;
import com.ola.olamera.camera.session.CameraCaptureCallbackHandlerWrapper;
import com.ola.olamera.camera.session.CameraCaptureResult;
import com.ola.olamera.camera.session.CameraSelector;
import com.ola.olamera.camera.session.PreviewConfig;
import com.ola.olamera.camera.session.SessionConfig;
import com.ola.olamera.camera.session.UserCameraSession;
import com.ola.olamera.render.DefaultCameraRender;
import com.ola.olamera.render.detector.IAlgTextureConsumer;
import com.ola.olamera.render.photo.ExportPhoto;
import com.ola.olamera.render.photo.SnapShotCommand;
import com.ola.olamera.render.view.CameraVideoView;

import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class ActivityCameraSession {


    //Run in main thread
    private UserCameraSession mUserCameraSession;

    private CameraVideoView mPreview;

    private final Context mContext;

    private CameraSelector.CameraLenFacing mCameraLenFacing = CameraSelector.CameraLenFacing.LEN_FACING_BACK;


    private boolean mIsActive;


    private final AtomicBoolean mDoingSnapshot = new AtomicBoolean(false);


    private boolean mFirstStartCamera = true;

    private final Object mSurfaceLock = new Object();
    @GuardedBy("mSurfaceLock")
    private DefaultCameraRender mDefaultCameraRender;


    public ActivityCameraSession(@NonNull Context context) {
        mContext = context;
    }


    public void onWindowCreate() {
        requestCameraSession(mCameraLenFacing);
    }

    public void onWindowActive() {
        mIsActive = true;
        processCameraSessionActive();
    }

    private void processCameraSessionActive() {
        long start = SystemClock.elapsedRealtime();
        if (mUserCameraSession == null || !mUserCameraSession.active()) {
            return;
        }
        if (!mFirstStartCamera) {
            return;
        }
        MutableLiveData<CameraState> cameraStateObservable = mUserCameraSession.getCameraState();

        if (cameraStateObservable == null) {
            return;
        }
        cameraStateObservable.observeForever(new Observer<CameraState>() {
            @Override
            public void onChanged(CameraState cameraState) {
                if (cameraState == CameraState.OPEN) {
                    Log.i("CameraMainWindow", "camera open time " + (SystemClock.elapsedRealtime() - start));
                    cameraStateObservable.removeObserver(this);
                }
            }
        });
        mFirstStartCamera = false;
    }


    public void onWindowInactive() {
        mIsActive = false;
        if (mUserCameraSession != null && mUserCameraSession.inactive()) {
//            mRepository.onInactive(this);
            mDoingSnapshot.set(false);
        }
    }

    public void onWindowDestroy() {
        onWindowInactive();
    }


    public void takePhoto() {
        if (mDoingSnapshot.get()) {
            return;
        }
        mDoingSnapshot.set(true);
        if (mPreview == null) {
            return;
        }
        long start = SystemClock.elapsedRealtime();


        SnapShotCommand snapShotCommand = new SnapShotCommand(0, 0, 1f, 1f, value -> {
            if (value == null || value.data == null) {
                return;
            }
            Bitmap bitmap = convertGLPixels2BitmapCache(value);

            mDoingSnapshot.set(false);


        });
        mPreview.snapshot(snapShotCommand);

    }


    public static Bitmap convertGLPixels2BitmapCache(ExportPhoto input) {
        if (input == null || !input.isValid()) {
            return null;
        }

        int width = input.width;
        int height = input.height;

        int totalLength = width * height * 4; //int = 4 bytes

        if (input.data == null || input.width == 0 || input.height == 0 || input.data.length < totalLength) {
            return null;
        }

        ByteBuffer buf = ByteBuffer.wrap(input.data);


        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.copyPixelsFromBuffer(buf);

        Matrix yFlip = new Matrix();
        //ARGB为从GL上读出来的数据，所以Y轴要上下翻转，GL坐标系和Bitmap坐标系问题
        yFlip.postScale(1, -1);
        return Bitmap.createBitmap(result, 0, 0, result.getWidth(), result.getHeight(), yFlip, false);
    }


    public void setCameraPreview(@NonNull CameraVideoView cameraPreview) {
        mPreview = cameraPreview;
//        mPreview.setOnAlgCpuDataReceiver(mPreviewDataCallback);
        setPreviewDataCallback(mPreviewDataCallback);
    }

    private final IPreviewSurfaceProvider mPreviewSurfaceProvider = new IPreviewSurfaceProvider() {

        @Override
        public Surface provide(@NonNull SurfaceRequest request) {
            synchronized (mSurfaceLock) {
                //释放当前surface texture
                onUseComplete(null);

                Log.i("CameraMainWindow", "provide surface texture");

                SurfaceTextureWrapper surfaceTextureWrapper = new SurfaceTextureWrapper(request.width, request.height);
                mDefaultCameraRender = new DefaultCameraRender(surfaceTextureWrapper, request.camera2Camera, request.repeatCaptureCallback);

                if (mPreview != null) {
                    mPreview.getRender().setCameraRender(mDefaultCameraRender);
                }

                return surfaceTextureWrapper.getSurface();
            }
        }

//        @Override
//        public Surface provide(int width, int height) {
//            synchronized (mSurfaceLock) {
//                //释放当前surface texture
//                onUseComplete(null);
//
//                Log.i("CameraMainWindow", "provide surface texture");
//
//                SurfaceTextureWrapper surfaceTextureWrapper = new SurfaceTextureWrapper(width, height);
//                mDefaultCameraRender = new DefaultCameraRender(surfaceTextureWrapper);
//
//                if (mPreview != null) {
//                    mPreview.getRender().setCameraRender(mDefaultCameraRender);
//                }
//
//                return surfaceTextureWrapper.getSurface();
//            }
//        }

        @Override
        public void onUseComplete(Surface surface) {
            synchronized (mSurfaceLock) {
                if (mDefaultCameraRender != null) {
                    Log.i("CameraMainWindow", "onReleaseUseComplete");
                    mDefaultCameraRender.destroySurface();
                    mDefaultCameraRender = null;
                }
            }
        }
    };


    private void requestCameraSession(CameraSelector.CameraLenFacing lenFacing) {

        Camera2Manager.getInstance(mContext)
                .addListener(result -> {
                    Size size = new Size(720 * 16 / 9, 720);
                    PreviewConfig previewConfig = new PreviewConfig(size.getWidth(), size.getHeight(), mPreview);
//                    previewConfig.setRepeatCaptureCallback(new CameraCaptureCallback() {
//                        @Override
//                        public void onCaptureCompleted(@NonNull CameraCaptureResult cameraCaptureResult) {
//                        }
//                    }, MainThreadExecutor.getInstance());


                    SessionConfig sessionConfig = new SessionConfig(previewConfig);
                    sessionConfig.setCameraErrorListener((cameraError, message) -> {
                        Log.e("CameraMainWindow", "ERROR!!!: " + cameraError + " : " + message);
                    }, null);

                    mUserCameraSession = result.requestCameraSession(sessionConfig);
                    mUserCameraSession
                            .setCameraSelector(new CameraSelector().setFacing(lenFacing));


                    if (mIsActive) {
                        processCameraSessionActive();
                    } else {
                        mUserCameraSession.inactive();
                    }

                }, MainThreadExecutor.getInstance());
    }


    public void enableFlash(boolean enable, CameraCaptureCallback callback) {
        if (mUserCameraSession != null) {
            mUserCameraSession.enableFlash(
                    enable,
                    callback != null ? new CameraCaptureCallbackHandlerWrapper(new Handler(Looper.getMainLooper()), callback) : null,
                    null);
        }
    }


    public void switchCamera(@NonNull CameraSelector.CameraLenFacing cameraLenFacing) {
        CameraSelector.CameraLenFacing lastCameraLenFacing = mCameraLenFacing;
        mCameraLenFacing = cameraLenFacing;

        if (mUserCameraSession != null
                && mUserCameraSession.isActive()
                && mCameraLenFacing != lastCameraLenFacing) {
            mUserCameraSession.inactive();
            requestCameraSession(mCameraLenFacing);
        }
    }


    private IAlgTextureConsumer.OnAlgCpuDataReceiver mPreviewDataCallback;

    public void setPreviewDataCallback(IAlgTextureConsumer.OnAlgCpuDataReceiver previewDataCallback) {
        mPreviewDataCallback = previewDataCallback;
//        if (mPreview != null) {
//            mPreview.setOnAlgCpuDataReceiver(mPreviewDataCallback);
//        }
    }
}
