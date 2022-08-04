package com.ola.olamera.camera.session;


import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.os.Build;
import android.view.Surface;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.ola.olamera.camera.camera.Camera2CameraImpl;
import com.ola.olamera.util.CameraLogger;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class CaptureControlCallback extends CameraCaptureSession.CaptureCallback {

    private long sLastLogTime = -1;
    private boolean mNeedLog = false;

    @Override
    public void onCaptureStarted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, long timestamp, long frameNumber) {
        super.onCaptureStarted(session, request, timestamp, frameNumber);

    }

    @Override
    public void onCaptureProgressed(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull CaptureResult partialResult) {
        super.onCaptureProgressed(session, request, partialResult);
    }

    @Override
    public void onCaptureCompleted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull TotalCaptureResult result) {
        super.onCaptureCompleted(session, request, result);
        if (System.currentTimeMillis() - sLastLogTime > 2000) {
            sLastLogTime = System.currentTimeMillis();
            CameraLogger.i(Camera2CameraImpl.TAG, "onCaptureCompleted " + request.getTag());
        }
    }

    @Override
    public void onCaptureFailed(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull CaptureFailure failure) {
        super.onCaptureFailed(session, request, failure);
        CameraLogger.i(Camera2CameraImpl.TAG, "onCaptureFailed " + request.getTag());
    }

    @Override
    public void onCaptureSequenceCompleted(@NonNull CameraCaptureSession session, int sequenceId, long frameNumber) {
        super.onCaptureSequenceCompleted(session, sequenceId, frameNumber);
    }

    @Override
    public void onCaptureSequenceAborted(@NonNull CameraCaptureSession session, int sequenceId) {
        super.onCaptureSequenceAborted(session, sequenceId);
        CameraLogger.i(Camera2CameraImpl.TAG, "onCaptureSequenceAborted " + sequenceId);
    }

    @Override
    public void onCaptureBufferLost(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull Surface target, long frameNumber) {
        super.onCaptureBufferLost(session, request, target, frameNumber);
    }
}
