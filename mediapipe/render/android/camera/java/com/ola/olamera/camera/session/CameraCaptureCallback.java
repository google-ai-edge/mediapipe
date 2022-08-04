package com.ola.olamera.camera.session;

import androidx.annotation.NonNull;

public abstract class CameraCaptureCallback {

    /**
     * This method is called when an image capture has fully completed and all the result metadata
     * is available.
     *
     * @param cameraCaptureResult The output metadata from the capture.
     */
    public void onCaptureCompleted(@NonNull CameraCaptureResult cameraCaptureResult) {
    }

    /**
     * This method is called instead of {@link #onCaptureCompleted} when the camera device failed to
     * produce a {@link CameraCaptureResult} for the request.
     *
     * @param failure The output failure from the capture, including the failure reason.
     */
    public void onCaptureFailed(@NonNull CameraCaptureFailure failure) {
    }
}