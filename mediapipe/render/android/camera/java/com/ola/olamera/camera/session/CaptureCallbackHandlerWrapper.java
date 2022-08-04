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

import java.util.concurrent.Executor;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
final class CaptureCallbackHandlerWrapper extends CameraCaptureSession.CaptureCallback {

    final CameraCaptureSession.CaptureCallback mWrappedCallback;
    private final Executor mExecutor;

    CaptureCallbackHandlerWrapper(@NonNull Executor executor,
                                  @NonNull CameraCaptureSession.CaptureCallback wrappedCallback) {
        mExecutor = executor;
        mWrappedCallback = wrappedCallback;
    }

    @Override
    public void onCaptureStarted(@NonNull final CameraCaptureSession session,
                                 @NonNull final CaptureRequest request, final long timestamp,
                                 final long frameNumber) {
        mExecutor.execute(() -> mWrappedCallback.onCaptureStarted(session, request, timestamp, frameNumber));
    }

    @Override
    public void onCaptureProgressed(@NonNull final CameraCaptureSession session,
                                    @NonNull final CaptureRequest request, @NonNull final CaptureResult partialResult) {
        mExecutor.execute(() -> mWrappedCallback.onCaptureProgressed(session, request, partialResult));
    }

    @Override
    public void onCaptureCompleted(@NonNull final CameraCaptureSession session,
                                   @NonNull final CaptureRequest request, @NonNull final TotalCaptureResult result) {
        mExecutor.execute(() -> mWrappedCallback.onCaptureCompleted(session, request, result));
    }

    @Override
    public void onCaptureFailed(@NonNull final CameraCaptureSession session,
                                @NonNull final CaptureRequest request, @NonNull final CaptureFailure failure) {
        mExecutor.execute(() -> mWrappedCallback.onCaptureFailed(session, request, failure));
    }

    @Override
    public void onCaptureSequenceCompleted(@NonNull final CameraCaptureSession session,
                                           final int sequenceId, final long frameNumber) {
        mExecutor.execute(() -> mWrappedCallback.onCaptureSequenceCompleted(session, sequenceId, frameNumber));
    }

    @Override
    public void onCaptureSequenceAborted(@NonNull final CameraCaptureSession session,
                                         final int sequenceId) {
        mExecutor.execute(() -> mWrappedCallback.onCaptureSequenceAborted(session, sequenceId));
    }

    @RequiresApi(24)
    @Override
    public void onCaptureBufferLost(@NonNull final CameraCaptureSession session,
                                    @NonNull final CaptureRequest request, @NonNull final Surface target,
                                    final long frameNumber) {
        mExecutor.execute(() -> mWrappedCallback.onCaptureBufferLost(session, request, target, frameNumber));
    }
}