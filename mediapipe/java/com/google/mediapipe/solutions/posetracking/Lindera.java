package com.google.mediapipe.solutions.posetracking;

import android.util.Log;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;

import androidx.appcompat.app.AppCompatActivity;

import com.google.mediapipe.solutioncore.CameraInput;
import com.google.mediapipe.solutioncore.SolutionGlSurfaceView;

public class Lindera {
    private ComputerVisionPlugin plugin;
    private static final int rotation = Surface.ROTATION_0;
    private PoseTracking poseTracking;

    // Live camera demo UI and camera components.
    private CameraInput cameraInput;
    private SolutionGlSurfaceView<PoseTrackingResult> glSurfaceView;

    public Lindera(ComputerVisionPlugin plugin){
        this.plugin = plugin;
    }

    public void initialize (ViewGroup computerVisionContainerView , AppCompatActivity appCompatActivity){
        setupStreamingModePipeline(computerVisionContainerView,appCompatActivity);
    }
    /** Sets up core workflow for streaming mode. */
    private void setupStreamingModePipeline(ViewGroup computerVisionContainerView,AppCompatActivity appCompatActivity) {
        // Initializes a new MediaPipe Face Detection solution instance in the streaming mode.
        poseTracking =
                new PoseTracking(
                        appCompatActivity,
                        PoseTrackingOptions.builder()
                                .setStaticImageMode(false)
                                .setLandmarkVisibility(true)
                                .setModelComplexity(0)
                                .setSmoothLandmarks(true)
                                .build());
        poseTracking.setErrorListener(
                (message, e) -> Log.e("Lindera", "MediaPipe Pose Tracking error:" + message));
        cameraInput = new CameraInput(appCompatActivity);

        cameraInput.setNewFrameListener(textureFrame -> poseTracking.send(textureFrame));


        // Initializes a new Gl surface view with a user-defined PoseTrackingResultGlRenderer.
        glSurfaceView =
                new SolutionGlSurfaceView<>(
                        appCompatActivity, poseTracking.getGlContext(), poseTracking.getGlMajorVersion());
        glSurfaceView.setSolutionResultRenderer(new PoseTrackingResultGlRenderer());
        glSurfaceView.setRenderInputImage(true);
        poseTracking.setResultListener(
                poseTrackingResult -> {
//                    logExampleKeypoint(poseTrackingResult);
                    glSurfaceView.setRenderData(poseTrackingResult);
                    glSurfaceView.requestRender();
                });

        // The runnable to start camera after the gl surface view is attached.
        // For video input source, videoInput.start() will be called when the video uri is available.
        glSurfaceView.post(()->{this.startCamera(appCompatActivity);});
        // Updates the preview layout.

        computerVisionContainerView.removeAllViewsInLayout();
        computerVisionContainerView.addView(glSurfaceView);
        glSurfaceView.setVisibility(View.VISIBLE);
        computerVisionContainerView.requestLayout();
    }

    private void startCamera(AppCompatActivity appCompatActivity) {
        cameraInput.getConverter(poseTracking.getGlContext()).setRotation(rotation);
        cameraInput.start(
                appCompatActivity,
                poseTracking.getGlContext(),
                CameraInput.CameraFacing.FRONT,
                glSurfaceView.getWidth(),
                glSurfaceView.getHeight());
    }

}
