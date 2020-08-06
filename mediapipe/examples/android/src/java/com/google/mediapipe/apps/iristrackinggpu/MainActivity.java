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

package com.google.mediapipe.apps.iristrackinggpu;

import android.graphics.SurfaceTexture;
import com.google.mediapipe.framework.Packet;
import java.util.HashMap;
import java.util.Map;

/** Main activity of MediaPipe iris tracking app. */
public class MainActivity extends com.google.mediapipe.apps.basic.MainActivity {
  private static final String TAG = "MainActivity";

  private static final String FOCAL_LENGTH_STREAM_NAME = "focal_length_pixel";

  @Override
  protected void onCameraStarted(SurfaceTexture surfaceTexture) {
    super.onCameraStarted(surfaceTexture);

    float focalLength = cameraHelper.getFocalLengthPixels();
    if (focalLength != Float.MIN_VALUE) {
      Packet focalLengthSidePacket = processor.getPacketCreator().createFloat32(focalLength);
      Map<String, Packet> inputSidePackets = new HashMap<>();
      inputSidePackets.put(FOCAL_LENGTH_STREAM_NAME, focalLengthSidePacket);
      processor.setInputSidePackets(inputSidePackets);
    }
  }
}
