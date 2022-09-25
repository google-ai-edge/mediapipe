// Copyright 2019 The MediaPipe Authors.
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

package com.google.mediapipe.apps.posetrackingsolutiongpu;

import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.graphics.SurfaceTexture;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;

import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.formats.proto.LocationDataProto;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.glutil.EglManager;
import com.google.mediapipe.solutioncore.CameraInput;
import com.google.mediapipe.solutioncore.SolutionGlSurfaceView;
import com.google.mediapipe.solutions.facedetection.FaceDetection;
import com.google.mediapipe.solutions.facedetection.FaceDetectionOptions;
import com.google.mediapipe.solutions.posetracking.PoseTracking;
import com.google.mediapipe.solutions.posetracking.PoseTrackingOptions;
import com.google.mediapipe.solutions.posetracking.PoseTrackingResult;

import java.util.ArrayList;


/** Main activity of MediaPipe basic app. */
public class MainActivity extends AppCompatActivity {
  private static final String TAG = "MainActivity";

  // Flips the camera-preview frames vertically by default, before sending them into FrameProcessor
  // to be processed in a MediaPipe graph, and flips the processed frames back when they are
  // displayed. This maybe needed because OpenGL represents images assuming the image origin is at
  // the bottom-left corner, whereas MediaPipe in general assumes the image origin is at the
  // top-left corner.
  // NOTE: use "flipFramesVertically" in manifest metadata to override this behavior.
  private static final boolean FLIP_FRAMES_VERTICALLY = true;

  // Number of output frames allocated in ExternalTextureConverter.
  // NOTE: use "converterNumBuffers" in manifest metadata to override number of buffers. For
  // example, when there is a FlowLimiterCalculator in the graph, number of buffers should be at
  // least `max_in_flight + max_in_queue + 1` (where max_in_flight and max_in_queue are used in
  // FlowLimiterCalculator options). That's because we need buffers for all the frames that are in
  // flight/queue plus one for the next frame from the camera.
  private static final int NUM_BUFFERS = 2;

  static {
    // Load all native libraries needed by the app.
    System.loadLibrary("mediapipe_jni");
    try {
      System.loadLibrary("opencv_java3");
    } catch (java.lang.UnsatisfiedLinkError e) {
      // Some example apps (e.g. template matching) require OpenCV 4.
      System.loadLibrary("opencv_java4");
    }
  }



  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(getContentViewLayoutResId());

    PoseTrackingOptions poseTrackingOptions = PoseTrackingOptions.builder()
            .setStaticImageMode(false).build();
    PoseTracking poseTracking = new PoseTracking(this,poseTrackingOptions);

    poseTracking.setErrorListener(
            (message, e) -> Log.e(TAG, "MediaPipe Face Detection error:" + message));
    CameraInput cameraInput = new CameraInput(this);


    cameraInput.setNewFrameListener(
            textureFrame -> poseTracking.send(textureFrame));
    SolutionGlSurfaceView<PoseTrackingResult> glSurfaceView =
            new SolutionGlSurfaceView<>(
                    this, poseTracking.getGlContext(), poseTracking.getGlMajorVersion());
    glSurfaceView.setSolutionResultRenderer(new PoseTrackingResultGlRenderer());
    glSurfaceView.setRenderInputImage(true);

    poseTracking.setResultListener(
            faceDetectionResult -> {
              if (faceDetectionResult.multiPoseTrackings().isEmpty()) {
                return;
              }
              LocationDataProto.LocationData locationData = faceDetectionResult
                      .multiPoseTrackings()
                      .get(0)
                      .getLocationData();
//                              .getRelativeKeypoints(FaceKeypoint.NOSE_TIP);
              Log.i(
                      TAG, locationData.toString());
//                      String.format(
//                              "MediaPipe Face Detection nose tip normalized coordinates (value range: [0, 1]): x=%f, y=%f",
//                              noseTip.getX(), noseTip.getY()));
              // Request GL rendering.
              glSurfaceView.setRenderData(faceDetectionResult);
              glSurfaceView.requestRender();
            });
            // The runnable to start camera after the GLSurfaceView is attached.
            glSurfaceView.post(
                    () ->
                            cameraInput.start(
                                    this,
                                    poseTracking.getGlContext(),
                                    CameraInput.CameraFacing.FRONT,
                                    glSurfaceView.getWidth(),
                                    glSurfaceView.getHeight()));
    glSurfaceView.setVisibility(View.VISIBLE);
    FrameLayout frameLayout = findViewById(R.id.preview_display_layout);
    frameLayout.removeAllViewsInLayout();
    frameLayout.addView(glSurfaceView);
    glSurfaceView.setVisibility(View.VISIBLE);
    frameLayout.requestLayout();
  }


  // Used to obtain the content view for this application. If you are extending this class, and
  // have a custom layout, override this method and return the custom layout.
  protected int getContentViewLayoutResId() {
    return R.layout.activity_main;
  }




}
