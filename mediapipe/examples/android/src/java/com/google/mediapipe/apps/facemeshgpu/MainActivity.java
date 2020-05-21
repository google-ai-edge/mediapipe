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

package com.google.mediapipe.apps.facemeshgpu;

import android.os.Bundle;
import android.util.Log;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Main activity of MediaPipe face mesh app. */
public class MainActivity extends com.google.mediapipe.apps.basic.MainActivity {
  private static final String TAG = "MainActivity";

  private static final String INPUT_NUM_FACES_SIDE_PACKET_NAME = "num_faces";
  private static final String OUTPUT_LANDMARKS_STREAM_NAME = "multi_face_landmarks";
  // Max number of faces to detect/process.
  private static final int NUM_FACES = 1;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    AndroidPacketCreator packetCreator = processor.getPacketCreator();
    Map<String, Packet> inputSidePackets = new HashMap<>();
    inputSidePackets.put(INPUT_NUM_FACES_SIDE_PACKET_NAME, packetCreator.createInt32(NUM_FACES));
    processor.setInputSidePackets(inputSidePackets);

    processor.addPacketCallback(
        OUTPUT_LANDMARKS_STREAM_NAME,
        (packet) -> {
          Log.d(TAG, "Received multi face landmarks packet.");
          List<NormalizedLandmarkList> multiFaceLandmarks =
              PacketGetter.getProtoVector(packet, NormalizedLandmarkList.parser());
          Log.d(
              TAG,
              "[TS:"
                  + packet.getTimestamp()
                  + "] "
                  + getMultiFaceLandmarksDebugString(multiFaceLandmarks));
        });
  }

  private static String getMultiFaceLandmarksDebugString(
      List<NormalizedLandmarkList> multiFaceLandmarks) {
    if (multiFaceLandmarks.isEmpty()) {
      return "No face landmarks";
    }
    String multiFaceLandmarksStr = "Number of faces detected: " + multiFaceLandmarks.size() + "\n";
    int faceIndex = 0;
    for (NormalizedLandmarkList landmarks : multiFaceLandmarks) {
      multiFaceLandmarksStr +=
          "\t#Face landmarks for face[" + faceIndex + "]: " + landmarks.getLandmarkCount() + "\n";
      int landmarkIndex = 0;
      for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
        multiFaceLandmarksStr +=
            "\t\tLandmark ["
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
      ++faceIndex;
    }
    return multiFaceLandmarksStr;
  }
}
