package com.ola.olamera.camera.camera;

import android.annotation.SuppressLint;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import android.os.Handler;
import android.os.SystemClock;
import android.text.TextUtils;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.lifecycle.MutableLiveData;

import com.ola.olamera.camera.anotaion.ExecutedBy;
import com.ola.olamera.camera.concurrent.HandlerScheduledExecutorService;
import com.ola.olamera.camera.sensor.DisplayOrientationDetector;
import com.ola.olamera.camera.session.ImageCapture;
import com.ola.olamera.camera.session.InnerImageCaptureCallback;
import com.ola.olamera.camera.session.RepeatCaptureRequestConfig;
import com.ola.olamera.camera.session.SessionConfig;
import com.ola.olamera.camera.session.SingleCaptureConfig;
import com.ola.olamera.camera.session.SyncCaptureSession;
import com.ola.olamera.util.CameraLogger;
import com.ola.olamera.util.CameraShould;
import com.ola.olamera.util.Preconditions;

import java.util.Collections;
import java.util.concurrent.Callable;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledFuture;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class Camera2CameraImpl {
    public static final String TAG = "AndroidCameraApi";

    private volatile InternalState mState = InternalState.INITIALIZED;

    private final String mCameraId;

    protected CameraDevice mCameraDevice;

    private final Executor mExecutor;
    private final Handler mHandler;

    private CameraCharacteristics mCameraCharacteristics;

    private final Camera2Info mCamera2Info;


    private final MutableLiveData<CameraState> mCameraPublishStateLD;

    private int mCameraDeviceError = ERROR_NONE;

    private CameraState mCameraPublishState;

    private final Camera2Control mControl;

    private ICameraErrorListener mErrorListener;

    //这个东西可能持有外部很多窗口,导致泄漏！！！　小心，尤其是里面的callback
    private SessionConfig mSessionConfig;

    private final CameraStateCallback mStateCallback;


    @NonNull
    private final CameraReopenMonitor mCameraReopenMonitor = new CameraReopenMonitor();


    enum InternalState {
        /**
         * Stable state once the camera has been constructed.
         *
         * <p>At this state the {@link CameraDevice} should be invalid, but threads should be still
         * in a valid state. Whenever a camera device is fully closed the camera should return to
         * this state.
         *
         * <p>After an error occurs the camera returns to this state so that the device can be
         * cleanly reopened.
         */
        INITIALIZED,
        /**
         * Camera is waiting for the camera to be available to open.
         *
         * <p>A camera may enter a pending state if the camera has been stolen by another process
         * or if the maximum number of available cameras is already open.
         *
         * <p>At the end of this state, the camera should move into the OPENING state.
         */
        PENDING_OPEN,
        /**
         * A transitional state where the camera device is currently opening.
         *
         * <p>At the end of this state, the camera should move into either the OPENED or CLOSING
         * state.
         */
        OPENING,
        /**
         * A stable state where the camera has been opened.
         *
         * <p>During this state the camera device should be valid. It is at this time a valid
         * capture session can be active. Capture requests should be issued during this state only.
         */
        OPENED,
        /**
         * A transitional state where the camera device is currently closing.
         *
         * <p>At the end of this state, the camera should move into the INITIALIZED state.
         */
        CLOSING,
        /**
         * A transitional state where the camera was previously closing, but not fully closed before
         * a call to open was made.
         *
         * <p>At the end of this state, the camera should move into one of two states. The OPENING
         * state if the device becomes fully closed, since it must restart the process of opening a
         * camera. The OPENED state if the device becomes opened, which can occur if a call to close
         * had been done during the OPENING state.
         */
        REOPENING,
        /**
         * A transitional state where the camera will be closing permanently.
         *
         * <p>At the end of this state, the camera should move into the RELEASED state.
         */
        @Deprecated
        RELEASING,
        /**
         * A stable state where the camera has been permanently closed.
         *
         * <p>During this state all resources should be released and all operations on the camera
         * will do nothing.
         */
        @Deprecated
        RELEASED
    }


    private final CameraManager mCameraManager;
    private final CameraAbilityCallback mCameraAbilityCallback;

    private SyncCaptureSession mCaptureSession;

    public Camera2CameraImpl(CameraManager cameraManager, String cameraId, Executor executor, Handler handler) {
        mExecutor = executor;
        mHandler = handler;
        mCameraManager = cameraManager;
        mCameraId = cameraId;
        mCameraAbilityCallback = new CameraAbilityCallback();
        mCameraManager.registerAvailabilityCallback(mCameraAbilityCallback, handler);
        mCamera2Info = new Camera2Info(cameraId);
        mStateCallback = new CameraStateCallback(new HandlerScheduledExecutorService(mHandler));
        mCameraPublishStateLD = new MutableLiveData<>();
        mControl = new Camera2Control(this);
        mErrorListener = new ComboCameraErrorListener(null);

    }


    private class CameraAbilityCallback extends CameraManager.AvailabilityCallback {

        private boolean mIsAvailable = true;

        @Override
        public void onCameraAvailable(@NonNull String cameraId) {
            if (!TextUtils.equals(cameraId, mCameraId)) {
                return;
            }

            CameraLogger.i(TAG, "Camera2.onCameraAvailable %s", cameraId);

            mIsAvailable = true;

            if (mState == InternalState.PENDING_OPEN) {
                openCameraDevices(false);
            }
        }

        public boolean isAvailable() {
            return mIsAvailable;
        }

        @Override
        public void onCameraUnavailable(@NonNull String cameraId) {
            if (!TextUtils.equals(cameraId, mCameraId)) {
                return;
            }
            CameraLogger.e(TAG, "Camera2.onCameraUnavailable %s", cameraId);


            //我就是要最高权限
            mIsAvailable = false;

            if (mState == InternalState.PENDING_OPEN) {
                //TODO 确定是否需要触发一下相机的关闭
                //TODO 要添加一下统计，看看多少场景下，一开始相机是不可用的
            }
        }
    }

    public void updateSessionConfig(SessionConfig sessionConfig) {
        mExecutor.execute(() -> {
            mSessionConfig = sessionConfig;
            mErrorListener = new ComboCameraErrorListener(Collections.singletonList(mSessionConfig.getCameraErrorListener()));
        });

    }

    public static final int ERROR_NONE = 0;

    private class CameraStateCallback extends CameraDevice.StateCallback {


        private ScheduledReopen mScheduledReopenRunnable;
        @SuppressWarnings("WeakerAccess") // synthetic accessor
        ScheduledFuture<?> mScheduledReopenHandle;

        private final HandlerScheduledExecutorService mScheduler;


        public CameraStateCallback(HandlerScheduledExecutorService scheduler) {
            mScheduler = scheduler;
        }

        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            CameraLogger.i(TAG, "CameraDevice.onOpen when %s ", mState);
            //This is called when the camera is open
            mCameraDeviceError = ERROR_NONE;
            mCameraDevice = camera;
            switch (mState) {
                case OPENING:
                case REOPENING:
                    setState(InternalState.OPENED);
                    mCaptureSession = new SyncCaptureSession(Camera2CameraImpl.this, mHandler);
                    mCaptureSession.open(camera, mSessionConfig);
                    break;
                case RELEASING:
                case CLOSING:
                    mCameraDevice.close();
                    mCameraDevice = null;
                    break;
                default:
                    throw new IllegalStateException(
                            "onOpened() should not be possible from state: " + mState);
            }

        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            CameraLogger.e(TAG, "CameraDevice.onDisconnected() " + mState);

            // Can be treated the same as camera in use beca`use in both situations the
            // CameraDevice needs to be closed before it can be safely reopened and used.
            onError(camera, CameraDevice.StateCallback.ERROR_CAMERA_IN_USE);
        }

        @Override
        public void onError(@NonNull CameraDevice camera, int error) {
            CameraLogger.e(TAG, "CameraDevice.onError( %s ) when  %s ", CameraLogger.getCameraErrorMessage(error), mState);

            onErrorInternal(camera, error);

            mErrorListener.onError(error, CameraLogger.getCameraErrorMessage(error));

        }


        @Override
        public void onClosed(@NonNull CameraDevice camera) {
            CameraLogger.i(TAG, "CameraDevice.onClosed");
            switch (mState) {
                case CLOSING:
                case RELEASING:
                    finishClose();
                    break;
                case REOPENING:
                    if (mCameraDeviceError == ERROR_NONE) {
                        openCameraDevices(false);
                    } else {
                        CameraLogger.e("Camera closed due to error: %s", CameraLogger.getCameraErrorMessage(mCameraDeviceError));
                        scheduleCameraReopen();
                    }
                    break;
            }
        }

        /**
         * Resets the camera reopen attempts monitor. This should be called when the camera open is
         * not triggered by a scheduled camera reopen, but rather by an explicit request.
         */
        @ExecutedBy("mExecutor")
        void resetReopenMonitor() {
            mCameraReopenMonitor.reset();
        }


        // Delay long enough to guarantee the app could have been backgrounded.
        // See ProcessLifecycleProvider for where this delay comes from.
        static final int REOPEN_DELAY_MS = 700;

        @ExecutedBy("mExecutor")
        boolean scheduleCameraReopen() {
            Preconditions.checkState(mScheduledReopenRunnable == null);
            Preconditions.checkState(mScheduledReopenHandle == null);

            if (mCameraReopenMonitor.canScheduleCameraReopen()) {
                mScheduledReopenRunnable = new ScheduledReopen(mExecutor);
                CameraLogger.i(TAG, "Attempting camera re-open in %dms: %s", REOPEN_DELAY_MS, mScheduledReopenRunnable);
                mScheduledReopenHandle = mScheduler.schedule(mScheduledReopenRunnable, REOPEN_DELAY_MS);
                return true;
            } else {
                CameraLogger.e(TAG,
                        "Camera reopening attempted for "
                                + CameraReopenMonitor.REOPEN_LIMIT_MS
                                + "ms without success.");
                setState(InternalState.INITIALIZED);


                mErrorListener.onError(CameraLogger.ERROR_TRY_REOPEN_ERROR,
                        CameraLogger.getCameraErrorMessage(CameraLogger.ERROR_TRY_REOPEN_ERROR) + ":" + CameraLogger.getCameraErrorMessage(mCameraDeviceError));
            }
            return false;
        }

        /**
         * Attempts to cancel reopen.
         *
         * <p>If successful, it is safe to finish closing the camera via {@link #finishClose()} as
         * a reopen will only be scheduled after {@link #onClosed(CameraDevice)} has been called.
         *
         * @return true if reopen was cancelled. False if no re-open was scheduled.
         */
        @ExecutedBy("mExecutor")
        boolean cancelScheduledReopen() {
            boolean cancelled = false;
            if (mScheduledReopenHandle != null) {
                // A reopen has been scheduled
                CameraLogger.i(TAG, "Cancelling scheduled re-open: " + mScheduledReopenRunnable);

                // Ensure the runnable doesn't try to open the camera if it has already
                // been pushed to the executor.
                mScheduledReopenRunnable.cancel();
                mScheduledReopenRunnable = null;

                // Un-schedule the runnable in case if hasn't run.
                mScheduledReopenHandle.cancel(/*mayInterruptIfRunning=*/false);
                mScheduledReopenHandle = null;

                cancelled = true;
            }

            return cancelled;
        }


    }


    // Should only be called once the camera device is actually closed.
    @ExecutedBy("mExecutor")
    private void finishClose() {
        Preconditions.checkState(mState == InternalState.RELEASING || mState == InternalState.CLOSING);
        mCameraDevice = null;

        if (mState == InternalState.CLOSING) {
            if (mCaptureSession != null) {
                mCaptureSession.forceRelease();
                mCaptureSession = null;
            }
            mSessionConfig = null;
            setState(InternalState.INITIALIZED);
        } else {
            // After a camera is released, it cannot be reopened, so we don't need to listen for
            // available camera changes.
            mCameraManager.unregisterAvailabilityCallback(mCameraAbilityCallback);

            setState(InternalState.RELEASED);
        }
    }


    public void open() {
        mExecutor.execute(this::openInternal);
    }


    @ExecutedBy("mExecutor")
    private void openInternal() {
        Preconditions.cameraThreadCheck();


        switch (mState) {
            case INITIALIZED:
                openCameraDevices(false);
                break;
            case CLOSING:
                setState(InternalState.REOPENING);
                break;
        }
    }

    @SuppressLint("MissingPermission")
    private void openCameraDevices(boolean fromScheduledCameraReopen) {
        Preconditions.cameraThreadCheck();

        if (!fromScheduledCameraReopen) {
            mStateCallback.resetReopenMonitor();
        }
        mStateCallback.cancelScheduledReopen();


        /**
         * As of API level 23, devices for which the AvailabilityCallback#onCameraUnavailable(String) callback has been called due to the device being in use by a lower-priority, background camera API client can still potentially be opened by calling this method when the calling camera API client has a higher priority than the current camera API client using this device.
         * In general, if the top, foreground activity is running within your application process, your process will be given the highest priority when accessing the camera, and this method will succeed even if the camera device is in use by another camera API client.
         * Any lower-priority application that loses control of the camera in this way will receive an CameraDevice.StateCallback.onDisconnected(CameraDevice) callback.
         * Opening the same camera ID twice in the same application will similarly cause the CameraDevice.StateCallback.onDisconnected(CameraDevice) callback being fired for the CameraDevice from the first open call and all ongoing tasks being droppped.
         */
        //By Design : false && mCameraAbilityCallback.isAvailable()
        if (false && !mCameraAbilityCallback.isAvailable()) {
            setState(InternalState.PENDING_OPEN);
            CameraLogger.e(TAG, "camera (%s) is inability", mCameraId);
            return;
        }

        setState(InternalState.OPENING);

        try {
            mCameraManager.openCamera(mCameraId, mStateCallback, mHandler);
        } catch (CameraAccessException e) {
            // Camera2 will call the onError() callback with the specific error code that
            // caused this failure. No need to do anything here.
            CameraLogger.e(TAG, "open camera error (CameraAccessException) %s ", e.getMessage());
        } catch (IllegalArgumentException e) {
            CameraLogger.e(TAG, "open camera error %s ", e.getMessage());
            String errorMessage = e.getMessage();
            /**
             * In the process of a error restart, the camera hardware did not recover so quickly
             * On Mi8 Android 11 ，devices will throws IllegalArgumentException(SupportCameraApi) when check
             * SupportCameraApi error rather than callback onError. This causes us to have only one
             * restart opportunity to take
             * effect.But this devices need 5s+ to recover.
             *
             * see https://cs.android.com/android/platform/superproject/+/master:frameworks/av/services/camera/libcameraservice/CameraService.cpp;drc=master;bpv=0;bpt=1;l=2324?hl=zh-cn
             */
            if (errorMessage != null
                    && errorMessage.contains("supportsCameraApi")
                    && errorMessage.contains("Unknown camera ID")
                    && mCameraDeviceError != ERROR_NONE) {
                if (mStateCallback.scheduleCameraReopen()) {
                    setState(InternalState.REOPENING);
                } else {
                    setState(InternalState.INITIALIZED);
                }
            } else {
                //Camera has not been open
                setState(InternalState.INITIALIZED);
            }
        } catch (Exception e) {
            CameraLogger.e(TAG, "open camera error %s ", e.getMessage());
            //Camera has not been open
            setState(InternalState.INITIALIZED);
        }
    }


    private void setState(InternalState state) {
        CameraState publicState;
        switch (state) {
            case INITIALIZED:
                publicState = CameraState.CLOSED;
                break;
            case PENDING_OPEN:
                publicState = CameraState.PENDING_OPEN;
                break;
            case OPENING:
            case REOPENING:
                publicState = CameraState.OPENING;
                break;
            case OPENED:
                publicState = CameraState.OPEN;
                break;
            case CLOSING:
                publicState = CameraState.CLOSING;
                break;
            case RELEASING:
                publicState = CameraState.RELEASING;
                break;
            case RELEASED:
                publicState = CameraState.RELEASED;
                break;
            default:
                throw new IllegalStateException("Unknown state: " + state);

        }
        CameraLogger.i(TAG, "camera state change publish_state(%s -> %s) , inner_state(%s -> %s) ",
                mCameraPublishState, publicState, mState, state);
        mState = state;
        mCameraPublishState = publicState;
        mCameraPublishStateLD.postValue(mCameraPublishState);
        mStateObserver.notifyStateChange(mCameraPublishState);
    }

    private final CameraStateObservable mStateObserver = new CameraStateObservable();

    /**
     * 立即更新的相机状态，更新时机为相机线程中，立即更新
     */
    public CameraStateObservable getCameraStateImmediatelyObservable() {
        return mStateObserver;
    }

    /**
     * UI线程更新的相机状态更新
     */
    public MutableLiveData<CameraState> getCameraStateObservable() {
        return mCameraPublishStateLD;
    }

    public CameraState getCameraState() {
        return mCameraPublishState;
    }

    @ExecutedBy("mExecutor")
    private void resetCaptureSessionInner() {
        SyncCaptureSession oldCaptureSession = mCaptureSession;
        if (oldCaptureSession != null) {
            oldCaptureSession.release();
        }
        mCaptureSession = new SyncCaptureSession(Camera2CameraImpl.this, mHandler);
    }

    public void resetCaptureSession() {
        mExecutor.execute(this::resetCaptureSessionInner);
    }


    public void doRepeatingCaptureAction(@NonNull RepeatCaptureRequestConfig config) {
        if (mCaptureSession != null) {
            mCaptureSession.doRepeatingCaptureAction(config);
        }
    }


    @ExecutedBy("mExecutor")
    private void onErrorInternal(@NonNull CameraDevice cameraDevice, int error) {
        Preconditions.cameraThreadCheck();
        // onError could be called before onOpened if there is an error opening the camera
        // during initialization, so keep track of it here.
        mCameraDevice = cameraDevice;
        mCameraDeviceError = error;

        switch (mState) {
            case RELEASING:
            case CLOSING:
                CameraLogger.e(TAG, String.format("CameraDevice.onError(): %s failed with %s while "
                                + "in %s state. Will finish closing camera.",
                        mCameraId, CameraLogger.getCameraErrorMessage(error), mState.name()));
                closeCamera();
                break;
            case OPENING:
            case OPENED:
            case REOPENING:
                CameraLogger.e(TAG, String.format("CameraDevice.onError(): %s failed with %s while "
                                + "in %s state. Will attempt recovering from error.",
                        mCameraId, CameraLogger.getCameraErrorMessage(error), mState.name()));
                handleErrorOnOpen(error);
                break;
            default:
                throw new IllegalStateException(
                        "onError() should not be possible from state: " + mState);
        }
    }


    @ExecutedBy("mExecutor")
    private void handleErrorOnOpen(int error) {
        Preconditions.cameraThreadCheck();
        CameraShould.beTrue(mState == InternalState.OPENING || mState == InternalState.OPENED || mState == InternalState.REOPENING,
                "Attempt to handle open error from non open state: " + mState);
        switch (error) {
            case CameraDevice.StateCallback.ERROR_CAMERA_DEVICE:
                // A fatal error occurred. The device should be reopened.
                // Fall through.
            case CameraDevice.StateCallback.ERROR_MAX_CAMERAS_IN_USE:
            case CameraDevice.StateCallback.ERROR_CAMERA_IN_USE:
                // Attempt to reopen the camera again. If there are no cameras available,
                // this will wait for the next available camera.
                CameraLogger.i(TAG, String.format("Attempt to reopen camera[%s] after error[%s]",
                        mCameraId, CameraLogger.getCameraErrorMessage(error)));
                reopenCameraAfterError();
                break;
            default:
                // TODO: Properly handle other errors. For now, we will close the camera.
                CameraLogger.e(
                        TAG,
                        "Error observed on open (or opening) camera device "
                                + mCameraId
                                + ": "
                                + CameraLogger.getCameraErrorMessage(error)
                                + " closing camera.");
                setState(InternalState.CLOSING);
                closeCamera();
                break;
        }


    }

    public void capture(@NonNull SingleCaptureConfig singleCaptureConfig,
                        @NonNull InnerImageCaptureCallback capturedCallback) {
        try {
            mExecutor.execute(() -> captureInner(singleCaptureConfig, capturedCallback));
        } catch (RejectedExecutionException e) {
            capturedCallback.onError(e);
        }
    }


    private void captureInner(SingleCaptureConfig singleCaptureConfig,
                              @NonNull InnerImageCaptureCallback capturedCallback) {
        CameraLogger.i(TAG, "begin issue capture request when %s", mState);
        switch (mState) {
            case OPENED:
                try {
                    mCaptureSession.capture(singleCaptureConfig, mCameraDevice, capturedCallback);
                } catch (Exception e) {
                    capturedCallback.onError(new IllegalStateException(e));
                }
                break;
            //TODO only test capture error case
//                mStateCallback.mScheduler.schedule(() -> {
//                    mTestCaptureError = true;
//                    mExecutor.execute(() -> onErrorInternal(mCameraDevice, CameraDevice.StateCallback.ERROR_CAMERA_DEVICE));
//                    return null;
//                }, 5000);
            default:
                capturedCallback.onError(new IllegalStateException("capture request when " + mState));
                break;
        }
    }


    @ExecutedBy("mExecutor")
    private void reopenCameraAfterError() {
        // After an error, we must close the current camera device before we can open a new
        // one. To accomplish this, we will close the current camera and wait for the
        // onClosed() callback to reopen the device. It is also possible that the device can
        // be closed immediately, so in that case we will open the device manually.
        Preconditions.checkState(mCameraDeviceError != ERROR_NONE,
                "Can only reopen camera device after error if the camera device is actually "
                        + "in an error state.");
        setState(InternalState.REOPENING);
        mCameraDevice.close();
    }


    public void close() {
        mExecutor.execute(this::closeInternal);
    }

    @ExecutedBy("mExecutor")
    private void closeInternal() {
        Preconditions.cameraThreadCheck();
        CameraLogger.i(TAG, "closeInternal when %s", mState);

        switch (mState) {
            case OPENED:
                setState(InternalState.CLOSING);
                closeCamera();
                break;
            case PENDING_OPEN:
                //等待有效相机，直接标志为关闭成功
                setState(InternalState.INITIALIZED);
                break;
            case OPENING:
            case REOPENING:
                setState(InternalState.CLOSING);
                boolean canFinish = mStateCallback.cancelScheduledReopen();
                if (canFinish) {
                    finishClose();
                }
                break;
        }

    }


    @ExecutedBy("mExecutor")
    private void closeCamera() {
        Preconditions.cameraThreadCheck();

        /**
         * opened状态之后的, 需要releaseCaptureSession
         *
         * @TestLog
         * TEST: 假设相机先关闭,然后延迟关闭{@link CameraCaptureSession#close()} 会发生 看看会发生什么
         * TEST_RESULT:{@link  CameraCaptureSession.StateCallback#onClosed(CameraCaptureSession)}还是会触发触发,所以依赖closed完成Surface回收没有问题
         *
         * TEST: OPPO R9m Android 5.1 这个机器,　{@link CameraCaptureSession#close()} 不会回调 {@link  CameraCaptureSession.StateCallback#onClosed(CameraCaptureSession)}还
         * 这样导致的我surface无法回收
         * TEST_RESULT:强制在相机关闭的时候，再forceRelease一下
         *
         */
        if (mCaptureSession != null) {
            mCaptureSession.release();
        }

        mCameraDevice.close();

        mCameraDevice = null;
    }

    public void release() {
        mExecutor.execute(this::releaseInner);
    }

    private void releaseInner() {
        CameraLogger.i(TAG, "release camera when %s", mState);

        mStateCallback.cancelScheduledReopen();

        if (mCaptureSession != null) {
            mCaptureSession.forceRelease();
            mCaptureSession = null;
        }

        mStateObserver.clear();

        mSessionConfig = null;
        mErrorListener = new ComboCameraErrorListener(null);
    }


    public @NonNull
    CameraCharacteristics getCameraCharacteristics() {
        return mCameraCharacteristics;
    }

    public void setCameraCharacteristics(CameraCharacteristics cameraCharacteristics) {
        mCameraCharacteristics = cameraCharacteristics;
        mCamera2Info.setCameraCharacteristics(mCameraCharacteristics);
    }

    public Camera2Info getCamera2Info() {
        return mCamera2Info;
    }

    public String getCameraId() {
        return mCameraId;
    }

    public Camera2Control getControl() {
        return mControl;
    }


    class CameraReopenMonitor {
        // Time limit since the first camera reopen attempt after which reopening the camera
        // should no longer be attempted.
        static final int REOPEN_LIMIT_MS = 5_000;
        static final int INVALID_TIME = -1;
        private long mFirstReopenTime = INVALID_TIME;

        boolean canScheduleCameraReopen() {
            final long now = SystemClock.uptimeMillis();

            // If it's the first attempt to reopen the camera
            if (mFirstReopenTime == INVALID_TIME) {
                mFirstReopenTime = now;
                return true;
            }

            final boolean hasReachedLimit = now - mFirstReopenTime >= REOPEN_LIMIT_MS;

            // If the limit has been reached, prevent further attempts to reopen the camera,
            // and reset [firstReopenTime].
            if (hasReachedLimit) {
                reset();
                return false;
            }

            return true;
        }

        void reset() {
            mFirstReopenTime = INVALID_TIME;
        }
    }

    /**
     * A {@link Runnable} which will attempt to reopen the camera after a scheduled delay.
     */
    class ScheduledReopen implements Callable<Boolean> {

        private final Executor mExecutor;
        private boolean mCancelled = false;

        ScheduledReopen(@NonNull Executor executor) {
            mExecutor = executor;
        }

        void cancel() {
            mCancelled = true;
        }


        @Override
        public Boolean call() throws Exception {
            mExecutor.execute(() -> {
                // Scheduled reopen may have been cancelled after execute(). Check to ensure
                // this is still the scheduled reopen.
                if (!mCancelled) {
                    Preconditions.checkState(mState == InternalState.REOPENING);
                    openCameraDevices(/*fromScheduledCameraReopen=*/true);
                }
            });
            return true;
        }
    }


    public void updateDisplayRotationDetector(DisplayOrientationDetector displayOrientationDetector) {
        mCamera2Info.setDisplayOrientationDetector(displayOrientationDetector);
    }


}