// Copyright 2020 Google LLC
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

package com.google.mediapipe.apps.instantmotiontracking;

import static java.lang.Math.max;

import android.content.ClipDescription;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.drawable.Drawable;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.InputMethodManager;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import com.bumptech.glide.Glide;
import com.bumptech.glide.load.resource.gif.GifDrawable;
import com.bumptech.glide.request.target.CustomTarget;
import com.bumptech.glide.request.transition.Transition;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.Packet;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This is the MainActivity that handles camera input, IMU sensor data acquisition
 * and sticker management for the InstantMotionTracking MediaPipe project.
 */
public class MainActivity extends com.google.mediapipe.apps.basic.MainActivity {
  private static final String TAG = "InstantMotionTrackingMainActivity";

  // Allows for automated packet transmission to graph
  private MediaPipePacketManager mediaPipePacketManager;

  private static final int TARGET_CAMERA_WIDTH = 960;
  private static final int TARGET_CAMERA_HEIGHT = 1280;
  private static final float TARGET_CAMERA_ASPECT_RATIO =
      (float) TARGET_CAMERA_WIDTH / (float) TARGET_CAMERA_HEIGHT;

  // Bounds for a single click (sticker anchor reset)
  private static final long CLICK_DURATION = 300; // ms
  private long clickStartMillis = 0;
  private ViewGroup viewGroup;
  // Contains dynamic layout of sticker data controller
  private LinearLayout buttonLayout;

  private ArrayList<StickerManager> stickerArrayList;
  // Current sticker being edited by user
  private StickerManager currentSticker;
  // Trip value used to determine sticker re-anchoring
  private static final String STICKER_SENTINEL_TAG = "sticker_sentinel";
  private int stickerSentinel = -1;

  // Define parameters for 'reactivity' of object
  private static final float ROTATION_SPEED = 5.0f;
  private static final float SCALING_FACTOR = 0.025f;

  // Parameters of device visual field for rendering system
  // (68 degrees, 4:3 for Pixel 4)
  // TODO : Make acquisition of this information automated
  private static final float VERTICAL_FOV_RADIANS = (float) Math.toRadians(68.0);
  private static final String FOV_SIDE_PACKET_TAG = "vertical_fov_radians";
  private static final String ASPECT_RATIO_SIDE_PACKET_TAG = "aspect_ratio";

  private static final String IMU_MATRIX_TAG = "imu_rotation_matrix";
  private static final int SENSOR_SAMPLE_DELAY = SensorManager.SENSOR_DELAY_FASTEST;
  private final float[] rotationMatrix = new float[9];

  private static final String STICKER_PROTO_TAG = "sticker_proto_string";
  // Assets for object rendering
  // All animation assets and tags for the first asset (1)
  private Bitmap asset3dTexture = null;
  private static final String ASSET_3D_TEXTURE = "robot/robot_texture.jpg";
  private static final String ASSET_3D_FILE = "robot/robot.obj.uuu";
  private static final String ASSET_3D_TEXTURE_TAG = "texture_3d";
  private static final String ASSET_3D_TAG = "asset_3d";
  // All GIF animation assets and tags
  private GIFEditText editText;
  private ArrayList<Bitmap> gifBitmaps = new ArrayList<>();
  private int gifCurrentIndex = 0;
  private Bitmap defaultGIFTexture = null;  // Texture sent if no gif available
  // last time the GIF was updated
  private long gifLastFrameUpdateMS = System.currentTimeMillis();
  private static final int GIF_FRAME_RATE = 20; // 20 FPS
  private static final String GIF_ASPECT_RATIO_TAG = "gif_aspect_ratio";
  private static final String DEFAULT_GIF_TEXTURE = "gif/default_gif_texture.jpg";
  private static final String GIF_FILE = "gif/gif.obj.uuu";
  private static final String GIF_TEXTURE_TAG = "gif_texture";
  private static final String GIF_ASSET_TAG = "gif_asset_name";

  private int cameraWidth = TARGET_CAMERA_WIDTH;
  private int cameraHeight = TARGET_CAMERA_HEIGHT;

  @Override
  protected Size cameraTargetResolution() {
    // Camera size is in landscape, so here we have (height, width)
    return new Size(TARGET_CAMERA_HEIGHT, TARGET_CAMERA_WIDTH);
  }

  @Override
  protected Size computeViewSize(int width, int height) {
    // Try to force aspect ratio of view size to match our target aspect ratio
    return new Size(height, (int) (height * TARGET_CAMERA_ASPECT_RATIO));
  }

  @Override
  protected void onPreviewDisplaySurfaceChanged(
      SurfaceHolder holder, int format, int width, int height) {
    super.onPreviewDisplaySurfaceChanged(holder, format, width, height);
    boolean isCameraRotated = cameraHelper.isCameraRotated();

    // cameraImageSize computation logic duplicated from base MainActivity
    Size viewSize = computeViewSize(width, height);
    Size cameraImageSize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);
    cameraWidth =
        isCameraRotated ? cameraImageSize.getHeight() : cameraImageSize.getWidth();
    cameraHeight =
        isCameraRotated ? cameraImageSize.getWidth() : cameraImageSize.getHeight();
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {

    super.onCreate(savedInstanceState);

    editText = findViewById(R.id.gif_edit_text);
    editText.setGIFCommitListener(
        new GIFEditText.GIFCommitListener() {
          @Override
          public void onGIFCommit(Uri contentUri, ClipDescription description) {
            // The application must have permission to access the GIF content
            grantUriPermission(
                "com.google.mediapipe.apps.instantmotiontracking",
                contentUri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION);
            // Set GIF frames from content URI
            setGIFBitmaps(contentUri.toString());
            // Close the keyboard upon GIF acquisition
            closeKeyboard();
          }
        });

    // Send loaded 3d render assets as side packets to graph
    prepareDemoAssets();
    AndroidPacketCreator packetCreator = processor.getPacketCreator();

    Map<String, Packet> inputSidePackets = new HashMap<>();
    inputSidePackets.put(ASSET_3D_TEXTURE_TAG,
      packetCreator.createRgbaImageFrame(asset3dTexture));
    inputSidePackets.put(ASSET_3D_TAG,
      packetCreator.createString(ASSET_3D_FILE));
    inputSidePackets.put(GIF_ASSET_TAG,
      packetCreator.createString(GIF_FILE));
    processor.setInputSidePackets(inputSidePackets);

    // Add frame listener to PacketManagement system
    mediaPipePacketManager = new MediaPipePacketManager();
    processor.setOnWillAddFrameListener(mediaPipePacketManager);

    // Send device properties to render objects via OpenGL
    Map<String, Packet> devicePropertiesSidePackets = new HashMap<>();
    // TODO: Note that if our actual camera stream resolution does not match the
    // requested aspect ratio, then we will need to update the value used for
    // this packet, or else tracking results will be off.
    devicePropertiesSidePackets.put(
        ASPECT_RATIO_SIDE_PACKET_TAG, packetCreator.createFloat32(TARGET_CAMERA_ASPECT_RATIO));
    devicePropertiesSidePackets.put(
        FOV_SIDE_PACKET_TAG, packetCreator.createFloat32(VERTICAL_FOV_RADIANS));
    processor.setInputSidePackets(devicePropertiesSidePackets);

    // Begin with 0 stickers in dataset
    stickerArrayList = new ArrayList<>();
    currentSticker = null;

    SensorManager sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
    List<Sensor> sensorList = sensorManager.getSensorList(Sensor.TYPE_ROTATION_VECTOR);
    sensorManager.registerListener(
        new SensorEventListener() {
          private final float[] rotMatFromVec = new float[9];

          @Override
          public void onAccuracyChanged(Sensor sensor, int accuracy) {}
          // Update procedure on sensor adjustment (phone changes orientation)

          @Override
          public void onSensorChanged(SensorEvent event) {
            // Get the Rotation Matrix from the Rotation Vector
            SensorManager.getRotationMatrixFromVector(rotMatFromVec, event.values);
            // AXIS_MINUS_X is used to remap the rotation matrix for left hand
            // rules in the MediaPipe graph
            SensorManager.remapCoordinateSystem(
                rotMatFromVec, SensorManager.AXIS_MINUS_X, SensorManager.AXIS_Y, rotationMatrix);
          }
        },
        (Sensor) sensorList.get(0),
        SENSOR_SAMPLE_DELAY);

    // Mechanisms for zoom, pinch, rotation, tap gestures
    buttonLayout = (LinearLayout) findViewById(R.id.button_layout);
    viewGroup = findViewById(R.id.preview_display_layout);
    viewGroup.setOnTouchListener(
        new View.OnTouchListener() {
          @Override
          public boolean onTouch(View v, MotionEvent event) {
            return manageUiTouch(event);
          }
        });
    refreshUi();
  }

  // Obtain our custom activity_main layout for InstantMotionTracking
  @Override
  protected int getContentViewLayoutResId() {
    return R.layout.instant_motion_tracking_activity_main;
  }

  // Manages a touch event in order to perform placement/rotation/scaling gestures
  // on virtual sticker objects.
  private boolean manageUiTouch(MotionEvent event) {
    if (currentSticker != null) {
      switch (event.getAction()) {
          // Detecting a single click for object re-anchoring
        case (MotionEvent.ACTION_DOWN):
          clickStartMillis = System.currentTimeMillis();
          break;
        case (MotionEvent.ACTION_UP):
          if (System.currentTimeMillis() - clickStartMillis <= CLICK_DURATION) {
            recordClick(event);
          }
          break;
        case (MotionEvent.ACTION_MOVE):
          // Rotation and Scaling are independent events and can occur simulataneously
          if (event.getPointerCount() == 2) {
            if (event.getHistorySize() > 1) {
              // Calculate user scaling of sticker
              float newScaleFactor = getNewScaleFactor(event, currentSticker.getScaleFactor());
              currentSticker.setScaleFactor(newScaleFactor);
              // calculate rotation (radians) for dynamic y-axis rotations
              float rotationIncrement = calculateRotationRadians(event);
              currentSticker.setRotation(currentSticker.getRotation() + rotationIncrement);
            }
          }
          break;
        default:
          // fall out
      }
    }
    return true;
  }

  // Returns a float value that is equal to the radians of rotation from a two-finger
  // MotionEvent recorded by the OnTouchListener.
  private static float calculateRotationRadians(MotionEvent event) {
    float tangentA =
        (float) Math.atan2(event.getY(1) - event.getY(0), event.getX(1) - event.getX(0));
    float tangentB =
        (float)
            Math.atan2(
                event.getHistoricalY(1, 0) - event.getHistoricalY(0, 0),
                event.getHistoricalX(1, 0) - event.getHistoricalX(0, 0));
    float angle = ((float) Math.toDegrees(tangentA - tangentB)) % 360f;
    angle += ((angle < -180f) ? +360f : ((angle > 180f) ? -360f : 0.0f));
    float rotationIncrement = (float) (Math.PI * ((angle * ROTATION_SPEED) / 180));
    return rotationIncrement;
  }

  // Returns a float value that is equal to the translation distance between
  // two-fingers that move in a pinch/spreading direction.
  private static float getNewScaleFactor(MotionEvent event, float currentScaleFactor) {
    double newDistance = getDistance(event.getX(0), event.getY(0), event.getX(1), event.getY(1));
    double oldDistance =
        getDistance(
            event.getHistoricalX(0, 0),
            event.getHistoricalY(0, 0),
            event.getHistoricalX(1, 0),
            event.getHistoricalY(1, 0));
    float signFloat =
        (newDistance < oldDistance)
            ? -SCALING_FACTOR
            : SCALING_FACTOR; // Are they moving towards each other?
    currentScaleFactor *= (1f + signFloat);
    return currentScaleFactor;
  }

  // Called if a single touch event is recorded on the screen and used to set the
  // new anchor position for the current sticker in focus.
  private void recordClick(MotionEvent event) {
    // First normalize our click position w.r.t. to the view display
    float x = (event.getX() / viewGroup.getWidth());
    float y = (event.getY() / viewGroup.getHeight());

    // MediaPipe can automatically crop our camera stream when displaying it to
    // our surface, which can throw off our touch point calulations. So we need
    // to replicate that logic here. See FrameScaleMode::kFillAndCrop usage in
    // gl_quad_renderer.cc for more details.
    float widthRatio = (float) viewGroup.getWidth() / (float) cameraWidth;
    float heightRatio = (float) viewGroup.getHeight() / (float) cameraHeight;

    float maxRatio = max(widthRatio, heightRatio);
    widthRatio /= maxRatio;
    heightRatio /= maxRatio;

    // Now we scale by the scale factors, and then reposition (since cropping
    // is always centered)
    x *= widthRatio;
    x += 0.5f * (1.0f - widthRatio);
    y *= heightRatio;
    y += 0.5f * (1.0f - heightRatio);

    // Finally, we can pass our adjusted x and y points to the StickerManager
    currentSticker.setAnchorCoordinate(x, y);
    stickerSentinel = currentSticker.getstickerId();
  }

  // Provided the X and Y coordinates of two points, the distance between them
  // will be returned.
  private static double getDistance(double x1, double y1, double x2, double y2) {
    return Math.hypot((y2 - y1), (x2 - x1));
  }

  // Called upon each button click, and used to populate the buttonLayout with the
  // current sticker data in addition to sticker controls (delete, remove, back).
  private void refreshUi() {
    if (currentSticker != null) { // No sticker in view
      buttonLayout.removeAllViews();
      ImageButton deleteSticker = new ImageButton(this);
      setControlButtonDesign(deleteSticker, R.drawable.baseline_clear_24);
      deleteSticker.setOnClickListener(
          new View.OnClickListener() {
            @Override
            public void onClick(View v) {
              if (currentSticker != null) {
                stickerArrayList.remove(currentSticker);
                currentSticker = null;
                refreshUi();
              }
            }
          });
      // Go to home sticker menu
      ImageButton goBack = new ImageButton(this);
      setControlButtonDesign(goBack, R.drawable.baseline_arrow_back_24);
      goBack.setOnClickListener(
          new View.OnClickListener() {
            @Override
            public void onClick(View v) {
              currentSticker = null;
              refreshUi();
            }
          });
      // Change sticker to next possible render
      ImageButton loopRender = new ImageButton(this);
      setControlButtonDesign(loopRender, R.drawable.baseline_loop_24);
      loopRender.setOnClickListener(
          new View.OnClickListener() {
            @Override
            public void onClick(View v) {
              currentSticker.setRender(currentSticker.getRender().iterate());
              refreshUi();
            }
          });
      buttonLayout.addView(deleteSticker);
      buttonLayout.addView(goBack);
      buttonLayout.addView(loopRender);

      // Add the GIF search option if current sticker is GIF
      if (currentSticker.getRender() == StickerManager.Render.GIF) {
        ImageButton gifSearch = new ImageButton(this);
        setControlButtonDesign(gifSearch, R.drawable.baseline_search_24);
        gifSearch.setOnClickListener(
            new View.OnClickListener() {
              @Override
              public void onClick(View v) {
                // Clear the text field to prevent text artifacts in GIF selection
                editText.setText("");
                // Open the Keyboard to allow user input
                openKeyboard();
              }
            });
        buttonLayout.addView(gifSearch);
      }
    } else {
      buttonLayout.removeAllViews();
      // Display stickers
      for (final StickerManager sticker : stickerArrayList) {
        final ImageButton stickerButton = new ImageButton(this);
        stickerButton.setOnClickListener(
            new View.OnClickListener() {
              @Override
              public void onClick(View v) {
                currentSticker = sticker;
                refreshUi();
              }
            });
        if (sticker.getRender() == StickerManager.Render.GIF) {
          setControlButtonDesign(stickerButton, R.drawable.asset_gif_preview);
        } else if (sticker.getRender() == StickerManager.Render.ASSET_3D) {
          setStickerButtonDesign(stickerButton, R.drawable.asset_3d_preview);
        }

        buttonLayout.addView(stickerButton);
      }
      ImageButton addSticker = new ImageButton(this);
      setControlButtonDesign(addSticker, R.drawable.baseline_add_24);
      addSticker.setOnClickListener(
          new View.OnClickListener() {
            @Override
            public void onClick(View v) {
              StickerManager newSticker = new StickerManager();
              stickerArrayList.add(newSticker);
              currentSticker = newSticker;
              refreshUi();
            }
          });
      ImageButton clearStickers = new ImageButton(this);
      setControlButtonDesign(clearStickers, R.drawable.baseline_clear_all_24);
      clearStickers.setOnClickListener(
          new View.OnClickListener() {
            @Override
            public void onClick(View v) {
              stickerArrayList.clear();
              refreshUi();
            }
          });

      buttonLayout.addView(addSticker);
      buttonLayout.addView(clearStickers);
    }
  }

  // Sets ImageButton UI for Control Buttons.
  private void setControlButtonDesign(ImageButton btn, int imageDrawable) {
    // btn.setImageDrawable(getResources().getDrawable(imageDrawable));
    btn.setImageDrawable(getDrawable(imageDrawable));
    btn.setBackgroundColor(Color.parseColor("#00ffffff"));
    btn.setColorFilter(Color.parseColor("#0494a4"));
    btn.setLayoutParams(new LinearLayout.LayoutParams(200, 200));
    btn.setPadding(25, 25, 25, 25);
    btn.setScaleType(ImageView.ScaleType.FIT_XY);
  }

  // Sets ImageButton UI for Sticker Buttons.
  private void setStickerButtonDesign(ImageButton btn, int imageDrawable) {
    btn.setImageDrawable(getDrawable(imageDrawable));
    btn.setBackground(getDrawable(R.drawable.circle_button));
    btn.setLayoutParams(new LinearLayout.LayoutParams(250, 250));
    btn.setPadding(25, 25, 25, 25);
    btn.setScaleType(ImageView.ScaleType.CENTER_INSIDE);
  }

  // Used to set ArrayList of Bitmap frames
  private void setGIFBitmaps(String gifUrl) {
    gifBitmaps = new ArrayList<>(); // Empty the bitmap array
    Glide.with(this)
        .asGif()
        .load(gifUrl)
        .into(
            new CustomTarget<GifDrawable>() {
              @Override
              public void onLoadCleared(Drawable placeholder) {}

              @Override
              public void onResourceReady(
                  GifDrawable resource, Transition<? super GifDrawable> transition) {
                try {
                  Object startConstant = resource.getConstantState();
                  Field frameManager = startConstant.getClass().getDeclaredField("frameLoader");
                  frameManager.setAccessible(true);
                  Object frameLoader = frameManager.get(startConstant);
                  Field decoder = frameLoader.getClass().getDeclaredField("gifDecoder");
                  decoder.setAccessible(true);

                  Object frameObject = (decoder.get(frameLoader));
                  for (int i = 0; i < resource.getFrameCount(); i++) {
                    frameObject.getClass().getMethod("advance").invoke(frameObject);
                    Bitmap bmp =
                        (Bitmap)
                            frameObject.getClass().getMethod("getNextFrame").invoke(frameObject);
                    gifBitmaps.add(flipHorizontal(bmp));
                  }
                } catch (Exception e) {
                  Log.e(TAG, "", e);
                }
              }
            });
  }

  // Bitmaps must be flipped due to native acquisition of frames from Android OS
  private static Bitmap flipHorizontal(Bitmap bmp) {
    Matrix matrix = new Matrix();
    // Flip Bitmap frames horizontally
    matrix.preScale(-1.0f, 1.0f);
    return Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
  }

  // Function that is continuously called in order to time GIF frame updates
  private void updateGIFFrame() {
    long millisPerFrame = 1000 / GIF_FRAME_RATE;
    if (System.currentTimeMillis() - gifLastFrameUpdateMS >= millisPerFrame) {
      // Update GIF timestamp
      gifLastFrameUpdateMS = System.currentTimeMillis();
      // Cycle through every possible frame and avoid a divide by 0
      gifCurrentIndex = gifBitmaps.isEmpty() ? 1 : (gifCurrentIndex + 1) % gifBitmaps.size();
    }
  }

  // Called once to popup the Keyboard via Android OS with focus set to editText
  private void openKeyboard() {
    editText.requestFocus();
    InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
    imm.showSoftInput(editText, InputMethodManager.SHOW_IMPLICIT);
  }

  // Called once to close the Keyboard via Android OS
  private void closeKeyboard() {
    View view = this.getCurrentFocus();
    if (view != null) {
      InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
      imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
    }
  }

  private void prepareDemoAssets() {
    // We render from raw data with openGL, so disable decoding preprocessing
    BitmapFactory.Options decodeOptions = new BitmapFactory.Options();
    decodeOptions.inScaled = false;
    decodeOptions.inDither = false;
    decodeOptions.inPremultiplied = false;

    try {
      InputStream inputStream = getAssets().open(DEFAULT_GIF_TEXTURE);
      defaultGIFTexture =
          flipHorizontal(
              BitmapFactory.decodeStream(inputStream, null /*outPadding*/, decodeOptions));
      inputStream.close();
    } catch (Exception e) {
      Log.e(TAG, "Error parsing object texture; error: ", e);
      throw new IllegalStateException(e);
    }

    try {
      InputStream inputStream = getAssets().open(ASSET_3D_TEXTURE);
      asset3dTexture = BitmapFactory.decodeStream(inputStream, null /*outPadding*/, decodeOptions);
      inputStream.close();
    } catch (Exception e) {
      Log.e(TAG, "Error parsing object texture; error: ", e);
      throw new IllegalStateException(e);
    }
  }

  private class MediaPipePacketManager implements FrameProcessor.OnWillAddFrameListener {
    @Override
    public void onWillAddFrame(long timestamp) {
      // set current GIF bitmap as default texture
      Bitmap currentGIFBitmap = defaultGIFTexture;
      // If current index is in bounds, display current frame
      if (gifCurrentIndex <= gifBitmaps.size() - 1) {
        currentGIFBitmap = gifBitmaps.get(gifCurrentIndex);
      }
      // Update to next GIF frame based on timing and frame rate
      updateGIFFrame();

      // Calculate and set the aspect ratio of the GIF
      float gifAspectRatio =
          (float) currentGIFBitmap.getWidth() / (float) currentGIFBitmap.getHeight();

      Packet stickerSentinelPacket = processor.getPacketCreator().createInt32(stickerSentinel);
      // Sticker sentinel value must be reset for next graph iteration
      stickerSentinel = -1;
      // Initialize sticker data protobufferpacket information
      Packet stickerProtoDataPacket =
          processor
              .getPacketCreator()
              .createSerializedProto(StickerManager.getMessageLiteData(stickerArrayList));
      // Define and set the IMU sensory information float array
      Packet imuDataPacket = processor.getPacketCreator().createFloat32Array(rotationMatrix);
      // Communicate GIF textures (dynamic texturing) to graph
      Packet gifTexturePacket = processor.getPacketCreator().createRgbaImageFrame(currentGIFBitmap);
      Packet gifAspectRatioPacket = processor.getPacketCreator().createFloat32(gifAspectRatio);
      processor
          .getGraph()
          .addConsumablePacketToInputStream(STICKER_SENTINEL_TAG, stickerSentinelPacket, timestamp);
      processor
          .getGraph()
          .addConsumablePacketToInputStream(STICKER_PROTO_TAG, stickerProtoDataPacket, timestamp);
      processor
          .getGraph()
          .addConsumablePacketToInputStream(IMU_MATRIX_TAG, imuDataPacket, timestamp);
      processor
          .getGraph()
          .addConsumablePacketToInputStream(GIF_TEXTURE_TAG, gifTexturePacket, timestamp);
      processor
          .getGraph()
          .addConsumablePacketToInputStream(GIF_ASPECT_RATIO_TAG, gifAspectRatioPacket, timestamp);
      stickerSentinelPacket.release();
      stickerProtoDataPacket.release();
      imuDataPacket.release();
      gifTexturePacket.release();
      gifAspectRatioPacket.release();
    }
  }
}
