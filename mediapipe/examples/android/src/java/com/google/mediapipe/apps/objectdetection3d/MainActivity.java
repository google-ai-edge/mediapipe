// Copyright 2020 The MediaPipe Authors.
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

import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.Packet;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/** Main activity of MediaPipe object detection 3D app. */
public class MainActivity extends com.google.mediapipe.apps.basic.MainActivity {
  private static final String TAG = "MainActivity";

  private static final String OBJ_TEXTURE = "texture.jpg";
  private static final String OBJ_FILE = "model.obj.uuu";
  private static final String BOX_TEXTURE = "classic_colors.png";
  private static final String BOX_FILE = "box.obj.uuu";

  // Assets.
  private Bitmap objTexture = null;
  private Bitmap boxTexture = null;

  // ApplicationInfo for retrieving metadata defined in the manifest.
  private ApplicationInfo applicationInfo;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    try {
      applicationInfo =
          getPackageManager().getApplicationInfo(getPackageName(), PackageManager.GET_META_DATA);
    } catch (NameNotFoundException e) {
      Log.e(TAG, "Cannot find application info: " + e);
    }
    // Get allowed object category.
    String categoryName = applicationInfo.metaData.getString("categoryName");
    // Get maximum allowed number of objects.
    int maxNumObjects = applicationInfo.metaData.getInt("maxNumObjects");
    float[] modelScale = parseFloatArrayFromString(
        applicationInfo.metaData.getString("modelScale"));
    float[] modelTransform = parseFloatArrayFromString(
        applicationInfo.metaData.getString("modelTransformation"));
    prepareDemoAssets();
    AndroidPacketCreator packetCreator = processor.getPacketCreator();
    Map<String, Packet> inputSidePackets = new HashMap<>();
    inputSidePackets.put("obj_asset_name", packetCreator.createString(OBJ_FILE));
    inputSidePackets.put("box_asset_name", packetCreator.createString(BOX_FILE));
    inputSidePackets.put("obj_texture", packetCreator.createRgbaImageFrame(objTexture));
    inputSidePackets.put("box_texture", packetCreator.createRgbaImageFrame(boxTexture));
    inputSidePackets.put("allowed_labels", packetCreator.createString(categoryName));
    inputSidePackets.put("max_num_objects", packetCreator.createInt32(maxNumObjects));
    inputSidePackets.put("model_scale", packetCreator.createFloat32Array(modelScale));
    inputSidePackets.put("model_transformation", packetCreator.createFloat32Array(modelTransform));
    processor.setInputSidePackets(inputSidePackets);
  }

  @Override
  protected Size cameraTargetResolution() {
    return new Size(1280, 960); // Prefer 4:3 aspect ratio (camera size is in landscape).
  }

  @Override
  protected Size computeViewSize(int width, int height) {
    return new Size(height, height * 3 / 4); // Prefer 3:4 aspect ratio.
  }

  @Override
  protected void onPreviewDisplaySurfaceChanged(
      SurfaceHolder holder, int format, int width, int height) {
    super.onPreviewDisplaySurfaceChanged(holder, format, width, height);

    boolean isCameraRotated = cameraHelper.isCameraRotated();
    Size cameraImageSize = cameraHelper.getFrameSize();
    processor.setOnWillAddFrameListener(
        (timestamp) -> {
          try {
            int cameraTextureWidth =
                isCameraRotated ? cameraImageSize.getHeight() : cameraImageSize.getWidth();
            int cameraTextureHeight =
                isCameraRotated ? cameraImageSize.getWidth() : cameraImageSize.getHeight();

            // Find limiting side and scale to 3:4 aspect ratio
            float aspectRatio = (float) cameraTextureWidth / (float) cameraTextureHeight;
            if (aspectRatio > 3.0 / 4.0) {
              // width too big
              cameraTextureWidth = (int) ((float) cameraTextureHeight * 3.0 / 4.0);
            } else {
              // height too big
              cameraTextureHeight = (int) ((float) cameraTextureWidth * 4.0 / 3.0);
            }
            Packet widthPacket = processor.getPacketCreator().createInt32(cameraTextureWidth);
            Packet heightPacket = processor.getPacketCreator().createInt32(cameraTextureHeight);

            try {
              processor.getGraph().addPacketToInputStream("input_width", widthPacket, timestamp);
              processor.getGraph().addPacketToInputStream("input_height", heightPacket, timestamp);
            } catch (RuntimeException e) {
              Log.e(
                  TAG,
                  "MediaPipeException encountered adding packets to input_width and input_height"
                      + " input streams.", e);
            }
            widthPacket.release();
            heightPacket.release();
          } catch (IllegalStateException ise) {
            Log.e(TAG, "Exception while adding packets to width and height input streams.");
          }
        });
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

  private static float[] parseFloatArrayFromString(String string) {
    String[] elements = string.split(",", -1);
    float[] array = new float[elements.length];
    for (int i = 0; i < elements.length; ++i) {
      array[i] = Float.parseFloat(elements[i]);
    }
    return array;
  }
}
