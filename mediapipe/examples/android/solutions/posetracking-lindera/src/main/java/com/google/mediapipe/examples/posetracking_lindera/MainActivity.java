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

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.afollestad.materialdialogs.MaterialDialog;
import com.google.common.collect.ArrayListMultimap;
import com.google.mediapipe.solutions.lindera.BodyJoints;
import com.google.mediapipe.solutions.lindera.CameraRotation;
import com.google.mediapipe.solutions.lindera.ComputerVisionPlugin;
import com.google.mediapipe.solutions.lindera.Lindera;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileStore;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.BiConsumer;
import java.util.function.Consumer;


/**
 * Main activity of MediaPipe Face Detection app.
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private Lindera lindera;
    private ComputerVisionPluginImpl plugin;
    private boolean isLinderaInitialized = false;
    private boolean isDetectionStarted = false;
    private boolean isLoggingStarted = false;

    // Live camera demo UI and camera components.


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        findViewById(R.id.button_set_model).setVisibility(View.GONE);
        findViewById(R.id.button_toggle_landmarks).setVisibility(View.GONE);
        findViewById(R.id.button_start_capture).setVisibility(View.GONE);
        setupLiveDemoUiComponents();
        plugin = new ComputerVisionPluginImpl();


        lindera = new Lindera(plugin);
        List<String> cameras = lindera.getAvailableCameras();
        // FRONT or BACK
        lindera.setCamera("FRONT");
        lindera.setCameraRotation(CameraRotation.AUTOMATIC);

        lindera.fpsHelper.onFpsUpdate = new Consumer<Double>() {
            @Override
            public void accept(Double fps) {
                String text = "FPS: "+String.format("%04.1f" ,fps);
                runOnUiThread(()-> {
                    TextView view = findViewById(R.id.fps_view);
                    view.setText(text);
                });
            }
        };

    }


    /**
     * Sets up the UI components for the live demo with camera input.
     */
    private void setupLiveDemoUiComponents() {

        Button startDetectionButton = findViewById(R.id.button_start_detection);
        Button toggleLandmarks = findViewById(R.id.button_toggle_landmarks);
        Button modelComplexity = findViewById(R.id.button_set_model);
        Button startCapture = findViewById(R.id.button_start_capture);
        FrameLayout frameLayout = findViewById(R.id.preview_display_layout);

        startDetectionButton.setOnClickListener(
                v -> {
//                    startCameraButton.setVisibility(View.GONE);
                    if (!isLinderaInitialized) {
                        modelLoadAsyncDialogue(()->{
                            lindera.initialize(frameLayout, MainActivity.this);
                            isLinderaInitialized = true;
                            startDetectionButton.setVisibility(View.GONE);
                            findViewById(R.id.button_set_model).setVisibility(View.VISIBLE);
                            findViewById(R.id.button_toggle_landmarks).setVisibility(View.VISIBLE);
                            findViewById(R.id.button_start_capture).setVisibility(View.VISIBLE);

                            updateLandmarkButtonText();
                            updateModelComplexityButtonText();
                        });


                    }
                    isDetectionStarted = !isDetectionStarted;


                });

        toggleLandmarks.setOnClickListener(
                v ->{
                    this.lindera.setLandmarksVisibility(!this.lindera.getLandmarkVisibility());
                    updateLandmarkButtonText();
                }
        );

        startCapture.setOnClickListener(v->{

            if (isLoggingStarted){
                startCapture.setText("Start Capture");
                isLoggingStarted = false;
                try {
                    JSONObject jsonObject = plugin.stopLoggingAndDumpOutput();
                    // save to downloads folder
                    final ContentValues values = new ContentValues();
                    values.put(MediaStore.MediaColumns.DISPLAY_NAME, "data.json");
                    values.put(MediaStore.MediaColumns.MIME_TYPE, "application/json");
                    values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS);

                    final ContentResolver resolver = getContentResolver();
                    Uri uri = null;
                    final Uri contentUri = MediaStore.Downloads.EXTERNAL_CONTENT_URI;
                    uri = resolver.insert(contentUri, values);
                    final OutputStream stream = resolver.openOutputStream(uri);
                    stream.write(jsonObject.toString().getBytes(StandardCharsets.UTF_8));
                    stream.close();
                    Toast.makeText(getApplicationContext(), "data.json saved to Downloads", Toast.LENGTH_LONG).show();


                } catch (JSONException | IllegalAccessException | IOException e) {
                    e.printStackTrace();
                    Toast.makeText(getApplicationContext(), "Failed to save data", Toast.LENGTH_LONG).show();

                }
            }
            else {
                ProgressBar pbar = new ProgressBar(this);
                frameLayout.addView(pbar);
                final Handler handler = new Handler(Looper.getMainLooper());
                startCapture.setEnabled(false);

                handler.postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        frameLayout.removeView(pbar);
                        startCapture.setEnabled(true);
                        startCapture.setText("Stop Capture");
                        plugin.startLogging();
                    }
                }, 5000);
                isLoggingStarted = true;
            }
        });

        modelComplexity.setOnClickListener(v->{
            int modelComplexityVal = lindera.getModelComplexity();

            new MaterialDialog.Builder(this)
                    .title("Choose Model Complexity")
                    .items(Arrays.asList("Lite","Full","Heavy"))
                    .itemsCallbackSingleChoice(modelComplexityVal, new MaterialDialog.ListCallbackSingleChoice() {
                        @Override
                        public boolean onSelection(MaterialDialog dialog, View view, int which, CharSequence text) {
                            /**
                             * If you use alwaysCallSingleChoiceCallback(), which is discussed below,
                             * returning false here won't allow the newly selected radio button to actually be selected.
                             **/
                            if (which != modelComplexityVal){
                                modelLoadAsyncDialogue(()-> {
                                    lindera.setModelComplexity(which);
                                    lindera.restartDetection();
                                    updateModelComplexityButtonText();
                                });
                            }
                            return true;
                        }
                    })
                    .positiveText("choose")
                    .show();
//                listItemsSingleChoice(R.array.my_items, initialSelection = 1);



        });

    }
    void updateLandmarkButtonText(){
        Button toggleLandmarks = findViewById(R.id.button_toggle_landmarks);

        if (this.lindera.getLandmarkVisibility()) {
            toggleLandmarks.setText("Show Landmarks (On)");
        }else{
            toggleLandmarks.setText("Show Landmarks (Off)");

        }
    }

    void updateModelComplexityButtonText(){
        String text = "Select Model ";
        switch (this.lindera.getModelComplexity()){
            case 0:
                text += "(lite)";
                break;
            case 1:
                text += "(full)";
                break;
            case 2:
                text += "(heavy)";
                break;

        }
        Button setModel = findViewById(R.id.button_set_model);
        setModel.setText(text);

    }

    void modelLoadAsyncDialogue(Runnable loader){
        ProgressBar pbar = new ProgressBar(this);
        MaterialDialog dialog = new MaterialDialog.Builder(this)
                .title("Loading Model")
                .customView(pbar, false)
                .build();
        dialog.show();
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Handler handler = new Handler(Looper.getMainLooper());

        executor.execute(new Runnable() {
            @Override
            public void run() {
                //Background work here
                handler.post(loader);
                dialog.dismiss();
            }
        });
    }









}
