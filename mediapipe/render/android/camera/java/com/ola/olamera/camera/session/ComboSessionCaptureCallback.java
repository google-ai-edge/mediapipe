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

import java.util.ArrayList;
import java.util.List;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
final class ComboSessionCaptureCallback
        extends CameraCaptureSession.CaptureCallback {
    private final List<CameraCaptureSession.CaptureCallback> mCallbacks = new ArrayList<>();

    ComboSessionCaptureCallback(List<CameraCaptureSession.CaptureCallback> callbacks) {
        for (CameraCaptureSession.CaptureCallback callback : callbacks) {
            if (callback == null) {
                continue;
            }
            mCallbacks.add(callback);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onCaptureBufferLost(
            @NonNull CameraCaptureSession session, @NonNull CaptureRequest request,
            @NonNull Surface surface, long frame) {
        for (CameraCaptureSession.CaptureCallback callback : mCallbacks) {
            callback.onCaptureBufferLost(session, request, surface, frame);
        }
    }

    @Override
    public void onCaptureCompleted(
            @NonNull CameraCaptureSession session, @NonNull CaptureRequest request,
            @NonNull TotalCaptureResult result) {
        for (CameraCaptureSession.CaptureCallback callback : mCallbacks) {
            callback.onCaptureCompleted(session, request, result);
        }
    }

    @Override
    public void onCaptureFailed(
            @NonNull CameraCaptureSession session, @NonNull CaptureRequest request,
            @NonNull CaptureFailure failure) {
        for (CameraCaptureSession.CaptureCallback callback : mCallbacks) {
            callback.onCaptureFailed(session, request, failure);
        }
    }

    @Override
    public void onCaptureProgressed(
            @NonNull CameraCaptureSession session, @NonNull CaptureRequest request,
            @NonNull CaptureResult partialResult) {
        for (CameraCaptureSession.CaptureCallback callback : mCallbacks) {
            callback.onCaptureProgressed(session, request, partialResult);
        }
    }

    @Override
    public void onCaptureSequenceAborted(@NonNull CameraCaptureSession session,
                                         int sequenceId) {
        for (CameraCaptureSession.CaptureCallback callback : mCallbacks) {
            callback.onCaptureSequenceAborted(session, sequenceId);
        }
    }

    @Override
    public void onCaptureSequenceCompleted(
            @NonNull CameraCaptureSession session, int sequenceId, long frame) {
        for (CameraCaptureSession.CaptureCallback callback : mCallbacks) {
            callback.onCaptureSequenceCompleted(session, sequenceId, frame);
        }
    }

    @Override
    public void onCaptureStarted(
            @NonNull CameraCaptureSession session, @NonNull CaptureRequest request,
            long timestamp, long frame) {
        for (CameraCaptureSession.CaptureCallback callback : mCallbacks) {
            callback.onCaptureStarted(session, request, timestamp, frame);
        }
    }
}