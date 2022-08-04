package com.ola.olamera.camera.session;
/*
 *
 *  Creation    :  20-11-23
 *  Author      : jiaming.wjm@
 */

import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.media.ImageReader;
import android.os.Build;
import android.os.Handler;
import android.util.Log;
import android.view.Surface;

import com.ola.olamera.camera.anotaion.ExecutedBy;
import com.ola.olamera.camera.camera.Camera2CameraImpl;
import com.ola.olamera.camera.camera.CameraState;
import com.ola.olamera.camera.imagereader.DeferrableImageReader;
import com.ola.olamera.camera.preview.IPreviewSurfaceProvider;
import com.ola.olamera.util.CameraLogger;
import com.ola.olamera.util.CameraShould;
import com.ola.olamera.util.TestOnly;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;

import androidx.annotation.GuardedBy;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class SyncCaptureSession {


    public static final String TAG = Camera2CameraImpl.TAG;
    /**
     * Lock on whether the camera is open or closed.
     */
    final Object mStateLock = new Object();

    @GuardedBy("mStateLock")
    private State mState = State.INITIALIZED;

    private CameraCaptureSession mCameraCaptureSession;

    private Camera2CameraImpl mCamera;

    private Handler mHandler;

    public SyncCaptureSession(Camera2CameraImpl camera, Handler handler) {
        mCamera = camera;
        mHandler = handler;
    }


    private CaptureRequest.Builder mCaptureRequestBuilder;

    @GuardedBy("mStateLock")
    private SessionConfig mSessionConfig;


    enum State {
        /**
         * The default state of the session before construction.
         */
//        UNINITIALIZED,
        /**
         * Stable state once the session has been constructed, but prior to the {@link
         * CameraCaptureSession} being opened.
         */
        INITIALIZED,
        /**
         * Transitional state when the {@link CameraCaptureSession} is in the process of being
         * opened.
         */
        OPENING,
        /**
         * SyncCaptureSession
         * Stable state where the {@link CameraCaptureSession} has been successfully opened. During
         * this state if a valid {@link SessionConfig} has been set then the {@link
         * CaptureRequest} will be issued.
         */
        OPENED,

        /**
         * Transitional state where the resources are being cleaned up.
         */
        RELEASING,
        /**
         * Terminal state where the session has been cleaned up. At this point the session should
         * not be used as nothing will happen in this state.
         */
        RELEASED
    }


    @TestOnly
    private long mFPSTestTime = 0;

    public void open(@NonNull CameraDevice cameraDevice, @NonNull SessionConfig config) {
        synchronized (mStateLock) {
            switch (mState) {
                case INITIALIZED:
                    mSessionConfig = config;
                    openCaptureSession(cameraDevice);
                    break;
                default:
                    CameraLogger.e(TAG, "Open not allowed in state: " + mState);
            }
        }
    }

    private void changeState(State newState) {
        synchronized (mStateLock) {
            mState = newState;
        }
    }

    private final StateControlCallback mStateController = new StateControlCallback();

    private final CaptureControlCallback mCaptureControlCallback = new CaptureControlCallback();

    private void openCaptureSession(@NonNull CameraDevice cameraDevice) {
        synchronized (mStateLock) {
            if (mSessionConfig == null) {
                return;
            }

            try {

                PreviewConfig previewConfig = mSessionConfig.getPreviewConfig();

                IPreviewSurfaceProvider.SurfaceRequest request = new IPreviewSurfaceProvider.SurfaceRequest();
                request.width = previewConfig.getActualWidth();
                request.height = previewConfig.getActualHeight();
                request.camera2Camera = mCamera;
                request.repeatCaptureCallback = previewConfig.getRepeatCaptureCallback();

                Surface previewSurface = previewConfig.getPreviewView().getSurfaceProvider().provide(request);

                //创建CaptureRequestBuilder，TEMPLATE_PREVIEW比表示预览请求
                mCaptureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                //设置Surface作为预览数据的显示界面

                ArrayList<Surface> surfaces = new ArrayList<>();
                surfaces.add(previewSurface);

                mCaptureRequestBuilder.addTarget(previewSurface);

                if (mSessionConfig.getPreviewConfig().getImageReaders() != null) {
                    for (DeferrableImageReader imageReader : mSessionConfig.getPreviewConfig().getImageReaders()) {
                        if (imageReader.unWrapper() != null) {
                            surfaces.add(imageReader.unWrapper().getSurface());
                            CameraLogger.i(Camera2CameraImpl.TAG, "add ImageReader Surface begin ( format:%d, %d*%d )", imageReader.getFormat(), imageReader.unWrapper().getWidth(), imageReader.unWrapper().getHeight());
                            mCaptureRequestBuilder.addTarget(imageReader.unWrapper().getSurface());
                        }
                    }
                }

                ImageCapture captureConfig = mSessionConfig.getImageCapture();
                if (captureConfig != null && captureConfig.getDeferrableImageReader() != null) {
                    surfaces.add(captureConfig.getDeferrableImageReader().unWrapper().getSurface());
                }

                CameraLogger.i(Camera2CameraImpl.TAG, "openCaptureSession begin ( preview: %d*%d )",
                        mSessionConfig.getPreviewConfig().getActualWidth(),
                        mSessionConfig.getPreviewConfig().getActualHeight());

                changeState(State.OPENING);

                cameraDevice.createCaptureSession(surfaces, mStateController, mHandler);
            } catch (CameraAccessException e) {
                CameraLogger.e(Camera2CameraImpl.TAG, "openCaptureSession error (%s) ", e.getMessage());
                CameraShould.fail("", e);
            }
        }
    }

    public void capture(SingleCaptureConfig singleCaptureConfig, @NonNull CameraDevice cameraDevice, @NonNull InnerImageCaptureCallback capturedCallback) {
        synchronized (mStateLock) {
            switch (mState) {
                case OPENED:
                    captureInner(singleCaptureConfig, cameraDevice, capturedCallback);
                    break;
                default:
                    if (capturedCallback != null) {
                        capturedCallback.onError(new IllegalStateException("can not capture image when " + mState));
                    }
            }
        }
    }

    private void captureInner(SingleCaptureConfig singleCaptureConfig, @NonNull CameraDevice cameraDevice,
                              @NonNull InnerImageCaptureCallback capturedCallback) {
        synchronized (mStateLock) {
            try {

                ImageCapture captureConfig = mSessionConfig.getImageCapture();

                if (captureConfig == null) {
                    throw new RuntimeException("init image capture config first");
                }


                // This is the CaptureRequest.Builder that we use to take a picture.
                final CaptureRequest.Builder captureBuilder =
                        cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);

                ImageReader actualImageReader = captureConfig.getDeferrableImageReader().unWrapper();
                captureBuilder.addTarget(actualImageReader.getSurface());


                int natureRotation = captureConfig.getDeferrableImageReader().getImageRotationHelper().getImageRotation();
                int sensorRotation = captureConfig.getDeferrableImageReader().getImageRotationHelper().getCameraSensorOrientation();

                int jpegRotation = sensorRotation;
                if (singleCaptureConfig != null && singleCaptureConfig.useNatureRotation()) {
                    jpegRotation = natureRotation;
                }
                captureBuilder.set(CaptureRequest.JPEG_ORIENTATION, jpegRotation);
                captureBuilder.set(CaptureRequest.JPEG_QUALITY, singleCaptureConfig != null ?
                        singleCaptureConfig.getJpegQuality() : 100);
                captureConfig.fillConfig(mCamera.getCameraCharacteristics(), captureBuilder);

                //选择对应的相机，初始化默认配置
                if (mSessionConfig.getSelectConfig() != null) {
                    mSessionConfig.getSelectConfig().fillConfig(mCamera.getCamera2Info(), captureBuilder);
                }

                //TODO NoBlockImageAnalyzer 相当于每次只有一个图片回掉,之前的回掉会给新触发的callback给替换掉
                captureConfig.getDeferrableImageReader().getNoBlockImageAnalyzer().setImageAnalyzer((image, cameraSensorRotation, imageRotation) -> {
                    CameraLogger.i(Camera2CameraImpl.TAG, "ImageCapture Success (%d * %d)", image.getWidth(), image.getHeight());
                    capturedCallback.onCaptureSuccess(image);
                });

                CameraCaptureSession.CaptureCallback innerCallback = new CameraCaptureSession.CaptureCallback() {

                    @Override
                    public void onCaptureFailed(@NonNull CameraCaptureSession session,
                                                @NonNull CaptureRequest request,
                                                @NonNull CaptureFailure failure) {
                        super.onCaptureFailed(session, request, failure);
                        capturedCallback.onError(new Exception("capture_error:" + failure.getReason()));
                    }

                    @Override
                    public void onCaptureStarted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, long timestamp, long frameNumber) {
                        capturedCallback.onCaptureStart();
                    }

                    @Override
                    public void onCaptureSequenceAborted(@NonNull CameraCaptureSession session, int sequenceId) {
                        super.onCaptureSequenceAborted(session, sequenceId);
                        capturedCallback.onError(new Exception("capture_error:onCaptureSequenceAborted"));

                    }
                };


                //TODO 目前只有闪光等设置了需要关闭Repeating模式，重新开启
                //但是这块改动影响了CameraCaptureState的逻辑，目前先不实现
    //            mCameraCaptureSession.stopRepeating();
    //            mCameraCaptureSession.abortCaptures();

                int id = mCameraCaptureSession.capture(captureBuilder.build(), innerCallback, mHandler);

                CameraLogger.i(Camera2CameraImpl.TAG, "send image capture request " + id);

            } catch (CameraAccessException e) {
                CameraLogger.e(Camera2CameraImpl.TAG, "openCaptureSession error (%s) ", e.getMessage());
                CameraShould.fail("", e);
                capturedCallback.onError(e);
            }
        }
    }


    /**
     * Callback for handling state changes to the {@link CameraCaptureSession}.
     *
     * <p>State changes are ignored once the CaptureSession has been closed.
     */
    final class StateControlCallback extends CameraCaptureSession.StateCallback {
        /**
         * {@inheritDoc}
         *
         * <p>Once the {@link CameraCaptureSession} has been configured then the capture request
         * will be immediately issued.
         */
        @Override
        public void onConfigured(@NonNull CameraCaptureSession session) {
            synchronized (mStateLock) {
                switch (mState) {
                    case INITIALIZED:
                    case OPENED:
                    case RELEASED:
                        throw new IllegalStateException(
                                "onConfigured() should not be possible in state: " + mState);
                    case OPENING:
                        if (mCamera.getCameraState() != CameraState.OPEN) {
                            //相机已经关闭了,直接关闭session
                            changeState(State.RELEASED);
                            return;
                        }
                        mState = State.OPENED;
                        changeState(State.OPENED);

                        mCameraCaptureSession = session;
                        CameraLogger.i("AndroidCameraApi", "Attempting to send capture request onConfigured");
                        issueRepeatingCaptureRequests(null);
                        break;
                    case RELEASING:
                        changeState(State.RELEASED);
                        session.close();
                        break;
                }
                CameraLogger.i(TAG, "CameraCaptureSession.onConfigured() mState=" + mState);
            }
        }

        @Override
        public void onReady(@NonNull CameraCaptureSession session) {
            synchronized (mStateLock) {
                switch (mState) {
                    case RELEASING:
                        if (mCameraCaptureSession == null) {
                            // No-op for releasing an unopened session.
                            break;
                        }
                        // The abortCaptures() called in release() has successfully finished.
                        mCameraCaptureSession.close();
                        break;
                    default:
                }
                CameraLogger.i(TAG, "CameraCaptureSession.onReady() " + mState);
            }
        }

        @Override
        public void onClosed(@NonNull CameraCaptureSession session) {
            synchronized (mStateLock) {

                if (mState == State.RELEASED) {
                    // If released then onClosed() has already been called, but it can be ignored
                    // since a session can be forceClosed.
                    return;
                }

                Log.d(TAG, "CameraCaptureSession.onClosed()");


                changeState(State.RELEASED);
                mCameraCaptureSession = null;

                if (mSessionConfig != null) {
                    mSessionConfig.getPreviewConfig().getPreviewView().getSurfaceProvider().onUseComplete(null);
                    mSessionConfig = null;
                }
            }
        }

        @Override
        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
            synchronized (mStateLock) {
                switch (mState) {
                    case INITIALIZED:
                    case OPENED:
                    case RELEASED:
                        throw new IllegalStateException(
                                "onConfiguredFailed() should not be possible in state: " + mState);
                    case OPENING:
                    case RELEASING:
                        changeState(State.RELEASING);
                        session.close();
                        break;
                }
                CameraLogger.i(TAG, "CameraCaptureSession.onConfiguredFailed() " + mState);
            }
        }
    }


    public void doRepeatingCaptureAction(@NonNull RepeatCaptureRequestConfig config) {
            mHandler.post(() -> {
                synchronized (mStateLock) {
                    switch (mState) {
                        case OPENED:
                            issueRepeatingCaptureRequests(config);
                            break;
                        default:
                            if (config.getCallback() != null) {
                                if (config.getCallbackExecutor() != null) {
                                    config.getCallbackExecutor().execute(() -> config.getCallback().onCaptureFailed(new CameraCaptureFailure(CameraCaptureFailure.Reason.ERROR)));
                                } else {
                                    config.getCallback().onCaptureFailed(new CameraCaptureFailure(CameraCaptureFailure.Reason.ERROR));
                                }
                                break;
                            }
                    }
                }
            });
    }


    private void issueRepeatingCaptureRequests(RepeatCaptureRequestConfig action) {
        synchronized (mStateLock) {
            if (mSessionConfig == null) {
                CameraLogger.e(TAG, "Skipping issueRepeatingCaptureRequests for no configuration case.");
                return;
            }

            try {

                //设置反复捕获数据的请求，这样预览界面就会一直有数据显示
                List<CameraCaptureSession.CaptureCallback> callbackList = new ArrayList<>();
                callbackList.add(mCaptureControlCallback);


                RepeatCaptureRequestConfig requestConfig = mSessionConfig.getPreviewConfig().getRepeatCaptureRequestConfig();
                requestConfig.fillConfig(mCamera.getCamera2Info(), mCaptureRequestBuilder);

                callbackList.add(convert2SystemApiCaptureCallback(requestConfig.getCallbackExecutor(), requestConfig.getCallback()));


                if (action != null) {
                    callbackList.add(convert2SystemApiCaptureCallback(action.getCallbackExecutor(), action.getCallback()));
                    action.fillConfig(mCamera.getCamera2Info(), mCaptureRequestBuilder);
                }

                //选择对应的相机，初始化默认配置
                if (mSessionConfig.getSelectConfig() != null) {
                    mSessionConfig.getSelectConfig().fillConfig(mCamera.getCamera2Info(), mCaptureRequestBuilder);
                }

                //创建捕获请求
                CaptureRequest request = mCaptureRequestBuilder.build();


                ComboSessionCaptureCallback comboSessionCaptureCallback = new ComboSessionCaptureCallback(callbackList);

                if (mCamera.getCameraState() == CameraState.OPEN) {
                    mCameraCaptureSession.setRepeatingRequest(request, comboSessionCaptureCallback, mHandler);
                }

            } catch (CameraAccessException e) {
                CameraLogger.e(Camera2CameraImpl.TAG, "CameraCaptureSession.openCaptureSession error (%s) ", e.getMessage());
                CameraShould.fail("", e);
            }
        }
    }


    /**
     * Releases the capture session.
     *
     * <p>This releases all of the sessions resources and should be called when ready to close the
     * camera.
     *
     * <p>Once a session is released it can no longer be opened again. After the session is released
     * all method calls on it do nothing.
     */
    @ExecutedBy("mHandler")
    public void release() {
        synchronized (mStateLock) {
            CameraLogger.i(Camera2CameraImpl.TAG, "CameraCaptureSession.releaseCaptureSession when (%s) %s ", mState, mCameraCaptureSession);
            switch (mState) {
                case OPENED:
                    if (mCameraCaptureSession != null) {
                        mCameraCaptureSession.close();
                    }
                    // Fall through
                case OPENING:
                    mState = State.RELEASING;
                    // Fall through
                case RELEASING:
                    break;
                case INITIALIZED:
                    mState = State.RELEASED;
                    // Fall through
                case RELEASED:
                    break;
            }
        }
    }

    @ExecutedBy("mHandler")
    public void forceRelease() {
        synchronized (mStateLock) {
            if (mSessionConfig != null) {
                mSessionConfig.getPreviewConfig()
                        .getPreviewView()
                        .getSurfaceProvider()
                        .onUseComplete(null);
                if (mSessionConfig.getImageCapture() != null
                        && mSessionConfig.getImageCapture().getDeferrableImageReader() != null) {
                    mSessionConfig.getImageCapture().getDeferrableImageReader().safeClose();
                }
                mSessionConfig = null;
            }
        }
    }

    private static CameraCaptureSession.CaptureCallback convert2SystemApiCaptureCallback(Executor executor, CameraCaptureCallback callback) {
        if (callback == null) {
            return null;
        }
        if (executor == null) {
            return new CaptureCallbackAdapter(callback);
        }


        return new CaptureCallbackHandlerWrapper(executor, new CaptureCallbackAdapter(callback));
    }

}
