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

package com.google.mediapipe.examples.posetracking_lindera;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;

import androidx.appcompat.app.AppCompatActivity;

import com.google.mediapipe.solutions.lindera.CameraRotation;
import com.google.mediapipe.solutions.lindera.ComputerVisionPlugin;
import com.google.mediapipe.solutions.lindera.Lindera;

import java.util.List;


/**
 * Main activity of MediaPipe Face Detection app.
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private Lindera lindera;
    private ComputerVisionPlugin plugin;
    private boolean isLinderaInitialized = false;
    private boolean isDetectionStarted = false;
    // Live camera demo UI and camera components.


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        disableRedundantUI();

        setupLiveDemoUiComponents();
        plugin = new ComputerVisionPluginImpl();
        lindera = new Lindera(plugin);
        List<String> cameras = lindera.getAvailableCameras();
        // FRONT or BACK
        lindera.setCamera("FRONT");
        lindera.setCameraRotation(CameraRotation.AUTOMATIC);

    }


    /**
     * Sets up the UI components for the live demo with camera input.
     */
    private void setupLiveDemoUiComponents() {

        Button startCameraButton = findViewById(R.id.button_start_camera);
        FrameLayout frameLayout = findViewById(R.id.preview_display_layout);

        startCameraButton.setOnClickListener(
                v -> {
//                    startCameraButton.setVisibility(View.GONE);
                    if (!isLinderaInitialized) {
                        lindera.initialize(frameLayout, MainActivity.this);
                        isLinderaInitialized = true;
                        startCameraButton.setText("STOP CAMERA");

                    } else {

                        if (isDetectionStarted) {
                            startCameraButton.setText(R.string.start_camera);


                            lindera.stopDetection();
                        } else {
                            lindera.startDetection();
                            startCameraButton.setText("STOP CAMERA");

                        }

                    }
                    isDetectionStarted = !isDetectionStarted;


                });
    }

    /**
     * Disables unecesary UI buttons
     */
    private void disableRedundantUI() {
        findViewById(R.id.button_load_picture).setVisibility(View.GONE);
        findViewById(R.id.button_load_video).setVisibility(View.GONE);

    }


}
