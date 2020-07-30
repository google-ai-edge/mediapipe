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

package com.google.mediapipe.apps.handtrackinggpu;

import android.os.Bundle;
import android.util.Log;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.framework.PacketGetter;
import com.google.protobuf.InvalidProtocolBufferException;

/** Main activity of MediaPipe hand tracking app. */
public class MainActivity extends com.google.mediapipe.apps.basic.MainActivity {
  private static final String TAG = "MainActivity";

  private static final String OUTPUT_HAND_PRESENCE_STREAM_NAME = "hand_presence";
  private static final String OUTPUT_LANDMARKS_STREAM_NAME = "hand_landmarks";

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    processor.addPacketCallback(
        OUTPUT_HAND_PRESENCE_STREAM_NAME,
        (packet) -> {
          Boolean handPresence = PacketGetter.getBool(packet);
          if (!handPresence) {
            Log.d(
                TAG,
                "[TS:" + packet.getTimestamp() + "] Hand presence is false, no hands detected.");
          }
        });

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
              Log.v(TAG, "[TS:" + packet.getTimestamp() + "] No hand landmarks.");
              return;
            }
            // Note: If hand_presence is false, these landmarks are useless.
            Log.v(
                TAG,
                "[TS:"
                    + packet.getTimestamp()
                    + "] #Landmarks for hand: "
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
