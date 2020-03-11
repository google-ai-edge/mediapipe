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

package com.google.mediapipe.apps.objectdetection3d;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.SurfaceTexture;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.glutil.EglManager;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/** Main activity of MediaPipe example apps. */
public class MainActivity extends AppCompatActivity {
  private static final String TAG = "MainActivity";

  private static final String BINARY_GRAPH_NAME = "objectdetection3d.binarypb";
  private static final String INPUT_VIDEO_STREAM_NAME = "input_video";
  private static final String OUTPUT_VIDEO_STREAM_NAME = "output_video";

  private static final String OBJ_TEXTURE = "texture.bmp";
  private static final String OBJ_FILE = "model.obj.uuu";
  private static final String BOX_TEXTURE = "classic_colors.png";
  private static final String BOX_FILE = "box.obj.uuu";

  private static final CameraHelper.CameraFacing CAMERA_FACING = CameraHelper.CameraFacing.BACK;

  // Flips the camera-preview frames vertically before sending them into FrameProcessor to be
  // processed in a MediaPipe graph, and flips the processed frames back when they are displayed.
  // This is needed because OpenGL represents images assuming the image origin is at the bottom-left
  // corner, whereas MediaPipe in general assumes the image origin is at top-left.
  private static final boolean FLIP_FRAMES_VERTICALLY = true;

  // Target resolution should be 4:3 for this application, as expected by the model and tracker.
  private static final Size TARGET_RESOLUTION = new Size(1280, 960);

  static {
    // Load all native libraries needed by the app.
    System.loadLibrary("mediapipe_jni");
    System.loadLibrary("opencv_java3");
  }

  // {@link SurfaceTexture} where the camera-preview frames can be accessed.
  private SurfaceTexture previewFrameTexture;
  // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
  private SurfaceView previewDisplayView;

  // Creates and manages an {@link EGLContext}.
  private EglManager eglManager;
  // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
  // frames onto a {@link Surface}.
  private FrameProcessor processor;
  // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
  // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
  private ExternalTextureConverter converter;

  // Handles camera access via the {@link CameraX} Jetpack support library.
  private CameraXPreviewHelper cameraHelper;

  // Assets.
  private Bitmap objTexture = null;
  private Bitmap boxTexture = null;

  Size cameraImageSize;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    previewDisplayView = new SurfaceView(this);
    setupPreviewDisplayView();

    // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
    // binary graphs.
    AndroidAssetUtil.initializeNativeAssetManager(this);

    eglManager = new EglManager(null);
    processor =
        new FrameProcessor(
            this,
            eglManager.getNativeContext(),
            BINARY_GRAPH_NAME,
            INPUT_VIDEO_STREAM_NAME,
            OUTPUT_VIDEO_STREAM_NAME);
    processor.getVideoSurfaceOutput().setFlipY(FLIP_FRAMES_VERTICALLY);

    prepareDemoAssets();
    AndroidPacketCreator packetCreator = processor.getPacketCreator();
    Map<String, Packet> inputSidePackets = new HashMap<>();
    inputSidePackets.put("obj_asset_name", packetCreator.createString(OBJ_FILE));
    inputSidePackets.put("box_asset_name", packetCreator.createString(BOX_FILE));
    inputSidePackets.put("obj_texture", packetCreator.createRgbaImageFrame(objTexture));
    inputSidePackets.put("box_texture", packetCreator.createRgbaImageFrame(boxTexture));
    processor.setInputSidePackets(inputSidePackets);

    PermissionHelper.checkAndRequestCameraPermissions(this);
  }

  @Override
  protected void onResume() {
    super.onResume();
    converter = new ExternalTextureConverter(eglManager.getContext());
    converter.setFlipY(FLIP_FRAMES_VERTICALLY);
    converter.setConsumer(processor);
    if (PermissionHelper.cameraPermissionsGranted(this)) {
      startCamera();
    }
  }

  @Override
  protected void onPause() {
    super.onPause();
    converter.close();
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
  }

  private void setupPreviewDisplayView() {
    previewDisplayView.setVisibility(View.GONE);
    ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
    viewGroup.addView(previewDisplayView);

    previewDisplayView
        .getHolder()
        .addCallback(
            new SurfaceHolder.Callback() {
              @Override
              public void surfaceCreated(SurfaceHolder holder) {
                processor.getVideoSurfaceOutput().setSurface(holder.getSurface());
              }

              @Override
              public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                // (Re-)Compute the ideal size of the camera-preview display (the area that the
                // camera-preview frames get rendered onto, potentially with scaling and rotation)
                // based on the size of the SurfaceView that contains the display.
                Size viewSize = new Size(height, height * 3 / 4); // Prefer 3:4 aspect ratio.
                Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);
                boolean isCameraRotated = cameraHelper.isCameraRotated();
                cameraImageSize = cameraHelper.getFrameSize();

                // Connect the converter to the camera-preview frames as its input (via
                // previewFrameTexture), and configure the output width and height as the computed
                // display size.
                converter.setSurfaceTextureAndAttachToGLContext(
                    previewFrameTexture,
                    isCameraRotated ? displaySize.getHeight() : displaySize.getWidth(),
                    isCameraRotated ? displaySize.getWidth() : displaySize.getHeight());
                processor.setOnWillAddFrameListener(
                    (timestamp) -> {
                      try {
                        int cameraTextureWidth =
                            isCameraRotated
                                ? cameraImageSize.getHeight()
                                : cameraImageSize.getWidth();
                        int cameraTextureHeight =
                            isCameraRotated
                                ? cameraImageSize.getWidth()
                                : cameraImageSize.getHeight();

                        // Find limiting side and scale to 3:4 aspect ratio
                        float aspectRatio =
                            (float) cameraTextureWidth / (float) cameraTextureHeight;
                        if (aspectRatio > 3.0 / 4.0) {
                          // width too big
                          cameraTextureWidth = (int) ((float) cameraTextureHeight * 3.0 / 4.0);
                        } else {
                          // height too big
                          cameraTextureHeight = (int) ((float) cameraTextureWidth * 4.0 / 3.0);
                        }
                        Packet widthPacket =
                            processor.getPacketCreator().createInt32(cameraTextureWidth);
                        Packet heightPacket =
                            processor.getPacketCreator().createInt32(cameraTextureHeight);

                        try {
                          processor
                              .getGraph()
                              .addPacketToInputStream("input_width", widthPacket, timestamp);
                          processor
                              .getGraph()
                              .addPacketToInputStream("input_height", heightPacket, timestamp);
                        } catch (Exception e) {
                          Log.e(
                              TAG,
                              "MediaPipeException encountered adding packets to width and height"
                                  + " input streams.");
                        }
                        widthPacket.release();
                        heightPacket.release();
                      } catch (IllegalStateException ise) {
                        Log.e(
                            TAG,
                            "Exception while adding packets to width and height input streams.");
                      }
                    });
              }

              @Override
              public void surfaceDestroyed(SurfaceHolder holder) {
                processor.getVideoSurfaceOutput().setSurface(null);
              }
            });
  }

  private void startCamera() {
    cameraHelper = new CameraXPreviewHelper();
    cameraHelper.setOnCameraStartedListener(
        surfaceTexture -> {
          previewFrameTexture = surfaceTexture;
          // Make the display view visible to start showing the preview. This triggers the
          // SurfaceHolder.Callback added to (the holder of) previewDisplayView.
          previewDisplayView.setVisibility(View.VISIBLE);
        });
    cameraHelper.startCamera(
        this, CAMERA_FACING, /*surfaceTexture=*/ null, /*targetSize=*/ TARGET_RESOLUTION);
    cameraImageSize = cameraHelper.getFrameSize();
  }

  private void prepareDemoAssets() {
    AndroidAssetUtil.initializeNativeAssetManager(this);
    // We render from raw data with openGL, so disable decoding preprocessing
    BitmapFactory.Options decodeOptions = new BitmapFactory.Options();
    decodeOptions.inScaled = false;
    decodeOptions.inDither = false;
    decodeOptions.inPremultiplied = false;

    try {
      InputStream inputStream = getAssets().open(OBJ_TEXTURE);
      objTexture = BitmapFactory.decodeStream(inputStream, null /*outPadding*/, decodeOptions);
      inputStream.close();
    } catch (Exception e) {
      Log.e(TAG, "Error parsing object texture; error: " + e);
      throw new IllegalStateException(e);
    }

    try {
      InputStream inputStream = getAssets().open(BOX_TEXTURE);
      boxTexture = BitmapFactory.decodeStream(inputStream, null /*outPadding*/, decodeOptions);
      inputStream.close();
    } catch (Exception e) {
      Log.e(TAG, "Error parsing box texture; error: " + e);
      throw new RuntimeException(e);
    }
  }
}
