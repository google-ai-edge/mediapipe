// Copyright 2021 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.examples.posetracking_camera;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;

import androidx.activity.result.ActivityResultLauncher;
import androidx.appcompat.app.AppCompatActivity;

import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.solutioncore.CameraInput;
import com.google.mediapipe.solutioncore.SolutionGlSurfaceView;
import com.google.mediapipe.solutions.posetracking.PoseTracking;
import com.google.mediapipe.solutions.posetracking.PoseTrackingOptions;
import com.google.mediapipe.solutions.posetracking.PoseTrackingResult;
import com.google.mediapipe.solutions.posetracking.PoseTrackingResultGlRenderer;


/**
 * Main activity of MediaPipe Face Detection app.
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int rotation = Surface.ROTATION_0;
    private PoseTracking poseTracking;


    private ActivityResultLauncher<Intent> videoGetter;
    // Live camera demo UI and camera components.
    private CameraInput cameraInput;

    private SolutionGlSurfaceView<PoseTrackingResult> glSurfaceView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        disableRedundantUI();

        setupLiveDemoUiComponents();

    }


    /**
     * Sets up the UI components for the live demo with camera input.
     */
    private void setupLiveDemoUiComponents() {

        Button startCameraButton = findViewById(R.id.button_start_camera);
        startCameraButton.setOnClickListener(
                v -> {
                    setupStreamingModePipeline();
                    startCameraButton.setVisibility(View.GONE);
                });
    }

    /**
     * Disables unecesary UI buttons
     */
    private void disableRedundantUI() {
        findViewById(R.id.button_load_picture).setVisibility(View.GONE);
        findViewById(R.id.button_load_video).setVisibility(View.GONE);

    }

    /**
     * Sets up core workflow for streaming mode.
     */
    private void setupStreamingModePipeline() {
        // Initializes a new MediaPipe Face Detection solution instance in the streaming mode.
        poseTracking =
                new PoseTracking(
                        this,
                        PoseTrackingOptions.builder()
                                .setStaticImageMode(false)
                                .setLandmarkVisibility(true)
                                .setModelComplexity(0)
                                .setSmoothLandmarks(true)
                                .build());
        poseTracking.setErrorListener(
                (message, e) -> Log.e(TAG, "MediaPipe Face Detection error:" + message));
        cameraInput = new CameraInput(this);

        cameraInput.setNewFrameListener(textureFrame -> poseTracking.send(textureFrame));


        // Initializes a new Gl surface view with a user-defined PoseTrackingResultGlRenderer.
        glSurfaceView =
                new SolutionGlSurfaceView<>(
                        this, poseTracking.getGlContext(), poseTracking.getGlMajorVersion());
        glSurfaceView.setSolutionResultRenderer(new PoseTrackingResultGlRenderer());
        glSurfaceView.setRenderInputImage(true);
        poseTracking.setResultListener(
                poseTrackingResult -> {
                    logExampleKeypoint(poseTrackingResult);
                    glSurfaceView.setRenderData(poseTrackingResult);
                    glSurfaceView.requestRender();
                });

        // The runnable to start camera after the gl surface view is attached.
        // For video input source, videoInput.start() will be called when the video uri is available.
        glSurfaceView.post(this::startCamera);

        // Updates the preview layout.
        FrameLayout frameLayout = findViewById(R.id.preview_display_layout);
        frameLayout.removeAllViewsInLayout();
        frameLayout.addView(glSurfaceView);
        glSurfaceView.setVisibility(View.VISIBLE);
        frameLayout.requestLayout();
    }

    private void startCamera() {
        cameraInput.getConverter(poseTracking.getGlContext()).setRotation(rotation);
        cameraInput.start(
                this,
                poseTracking.getGlContext(),
                CameraInput.CameraFacing.FRONT,
                glSurfaceView.getWidth(),
                glSurfaceView.getHeight());
    }


    private void logExampleKeypoint(
            PoseTrackingResult result) {
        if (result.multiPoseLandmarks().isEmpty()) {
            return;
        }
        LandmarkProto.Landmark exampleLandmark = result.multiPoseLandmarks().get(PoseTrackingResult.NOSE);
        Log.i(
                TAG,
                String.format(
                        "Pose Landmark Landmark of Nose: x=%f, y=%f, z=%f",
                        exampleLandmark.getX(), exampleLandmark.getY(), exampleLandmark.getZ()));


    }
}
