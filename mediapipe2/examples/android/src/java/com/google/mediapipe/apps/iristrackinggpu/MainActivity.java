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
import android.os.Bundle;
import android.util.Log;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.HashMap;
import java.util.Map;

/** Main activity of MediaPipe iris tracking app. */
public class MainActivity extends com.google.mediapipe.apps.basic.MainActivity {
  private static final String TAG = "MainActivity";

  private static final String FOCAL_LENGTH_STREAM_NAME = "focal_length_pixel";
  private static final String OUTPUT_LANDMARKS_STREAM_NAME = "face_landmarks_with_iris";

  private boolean haveAddedSidePackets = false;

  @Override
  protected void onCameraStarted(SurfaceTexture surfaceTexture) {
    super.onCameraStarted(surfaceTexture);

    // onCameraStarted gets called each time the activity resumes, but we only want to do this once.
    if (!haveAddedSidePackets) {
      float focalLength = cameraHelper.getFocalLengthPixels();
      if (focalLength != Float.MIN_VALUE) {
        Packet focalLengthSidePacket = processor.getPacketCreator().createFloat32(focalLength);
        Map<String, Packet> inputSidePackets = new HashMap<>();
        inputSidePackets.put(FOCAL_LENGTH_STREAM_NAME, focalLengthSidePacket);
        processor.setInputSidePackets(inputSidePackets);
      }
      haveAddedSidePackets = true;
    }
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    // To show verbose logging, run:
    // adb shell setprop log.tag.MainActivity VERBOSE
    if (Log.isLoggable(TAG, Log.VERBOSE)) {
      processor.addPacketCallback(
          OUTPUT_LANDMARKS_STREAM_NAME,
          (packet) -> {
            byte[] landmarksRaw = PacketGetter.getProtoBytes(packet);
            try {
              NormalizedLandmarkList landmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
              if (landmarks == null) {
                Log.v(TAG, "[TS:" + packet.getTimestamp() + "] No landmarks.");
                return;
              }
              Log.v(
                  TAG,
                  "[TS:"
                      + packet.getTimestamp()
                      + "] #Landmarks for face (including iris): "
                      + landmarks.getLandmarkCount());
              Log.v(TAG, getLandmarksDebugString(landmarks));
            } catch (InvalidProtocolBufferException e) {
              Log.e(TAG, "Couldn't Exception received - " + e);
              return;
            }
          });
    }
  }

  private static String getLandmarksDebugString(NormalizedLandmarkList landmarks) {
    int landmarkIndex = 0;
    String landmarksString = "";
    for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
      landmarksString +=
          "\t\tLandmark["
              + landmarkIndex
              + "]: ("
              + landmark.getX()
              + ", "
              + landmark.getY()
              + ", "
              + landmark.getZ()
              + ")\n";
      ++landmarkIndex;
    }
    return landmarksString;
  }
}
