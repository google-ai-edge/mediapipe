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

package com.google.mediapipe.apps.faceeffect;

import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.GestureDetector;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewGroup.LayoutParams;
import android.widget.RelativeLayout;
import android.widget.TextView;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.modules.facegeometry.FaceGeometryProto.FaceGeometry;
import com.google.mediapipe.formats.proto.MatrixDataProto.MatrixData;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Main activity of MediaPipe face mesh app. */
public class MainActivity extends com.google.mediapipe.apps.basic.MainActivity {
  private static final String TAG = "MainActivity";

  // Side packet / stream names.
  private static final String USE_FACE_DETECTION_INPUT_SOURCE_INPUT_SIDE_PACKET_NAME =
      "use_face_detection_input_source";
  private static final String SELECTED_EFFECT_ID_INPUT_STREAM_NAME = "selected_effect_id";
  private static final String OUTPUT_FACE_GEOMETRY_STREAM_NAME = "multi_face_geometry";

  private static final String EFFECT_SWITCHING_HINT_TEXT = "Tap to switch between effects!";

  private static final boolean USE_FACE_DETECTION_INPUT_SOURCE = false;
  private static final int MATRIX_TRANSLATION_Z_INDEX = 14;

  private static final int SELECTED_EFFECT_ID_AXIS = 0;
  private static final int SELECTED_EFFECT_ID_FACEPAINT = 1;
  private static final int SELECTED_EFFECT_ID_GLASSES = 2;

  private final Object effectSelectionLock = new Object();
  private int selectedEffectId;

  private View effectSwitchingHintView;
  private GestureDetector tapGestureDetector;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    // Add an effect switching hint view to the preview layout.
    effectSwitchingHintView = createEffectSwitchingHintView();
    effectSwitchingHintView.setVisibility(View.INVISIBLE);
    ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
    viewGroup.addView(effectSwitchingHintView);

    // By default, render the axis effect for the face detection input source and the glasses effect
    // for the face landmark input source.
    if (USE_FACE_DETECTION_INPUT_SOURCE) {
      selectedEffectId = SELECTED_EFFECT_ID_AXIS;
    } else {
      selectedEffectId = SELECTED_EFFECT_ID_GLASSES;
    }

    // Pass the USE_FACE_DETECTION_INPUT_SOURCE flag value as an input side packet into the graph.
    Map<String, Packet> inputSidePackets = new HashMap<>();
    inputSidePackets.put(
        USE_FACE_DETECTION_INPUT_SOURCE_INPUT_SIDE_PACKET_NAME,
        processor.getPacketCreator().createBool(USE_FACE_DETECTION_INPUT_SOURCE));
    processor.setInputSidePackets(inputSidePackets);

    // This callback demonstrates how the output face geometry packet can be obtained and used
    // in an Android app. As an example, the Z-translation component of the face pose transform
    // matrix is logged for each face being equal to the approximate distance away from the camera
    // in centimeters.
    processor.addPacketCallback(
        OUTPUT_FACE_GEOMETRY_STREAM_NAME,
        (packet) -> {
          effectSwitchingHintView.post(
              () ->
                  effectSwitchingHintView.setVisibility(
                      USE_FACE_DETECTION_INPUT_SOURCE ? View.INVISIBLE : View.VISIBLE));

          Log.d(TAG, "Received a multi face geometry packet.");
          List<FaceGeometry> multiFaceGeometry =
              PacketGetter.getProtoVector(packet, FaceGeometry.parser());

          StringBuilder approxDistanceAwayFromCameraLogMessage = new StringBuilder();
          for (FaceGeometry faceGeometry : multiFaceGeometry) {
            if (approxDistanceAwayFromCameraLogMessage.length() > 0) {
              approxDistanceAwayFromCameraLogMessage.append(' ');
            }
            MatrixData poseTransformMatrix = faceGeometry.getPoseTransformMatrix();
            approxDistanceAwayFromCameraLogMessage.append(
                -poseTransformMatrix.getPackedData(MATRIX_TRANSLATION_Z_INDEX));
          }

          Log.d(
              TAG,
              "[TS:"
                  + packet.getTimestamp()
                  + "] size = "
                  + multiFaceGeometry.size()
                  + "; approx. distance away from camera in cm for faces = ["
                  + approxDistanceAwayFromCameraLogMessage
                  + "]");
        });

    // Alongside the input camera frame, we also send the `selected_effect_id` int32 packet to
    // indicate which effect should be rendered on this frame.
    processor.setOnWillAddFrameListener(
        (timestamp) -> {
          Packet selectedEffectIdPacket = null;
          try {
            synchronized (effectSelectionLock) {
              selectedEffectIdPacket = processor.getPacketCreator().createInt32(selectedEffectId);
            }

            processor
                .getGraph()
                .addPacketToInputStream(
                    SELECTED_EFFECT_ID_INPUT_STREAM_NAME, selectedEffectIdPacket, timestamp);
          } catch (RuntimeException e) {
            Log.e(
                TAG, "Exception while adding packet to input stream while switching effects: " + e);
          } finally {
            if (selectedEffectIdPacket != null) {
              selectedEffectIdPacket.release();
            }
          }
        });

    // We use the tap gesture detector to switch between face effects. This allows users to try
    // multiple pre-bundled face effects without a need to recompile the app.
    tapGestureDetector =
        new GestureDetector(
            this,
            new GestureDetector.SimpleOnGestureListener() {
              @Override
              public void onLongPress(MotionEvent event) {
                switchEffect();
              }

              @Override
              public boolean onSingleTapUp(MotionEvent event) {
                switchEffect();
                return true;
              }

              private void switchEffect() {
                // Avoid switching the Axis effect for the face detection input source.
                if (USE_FACE_DETECTION_INPUT_SOURCE) {
                  return;
                }

                // Looped effect order: glasses -> facepaint -> axis -> glasses -> ...
                synchronized (effectSelectionLock) {
                  switch (selectedEffectId) {
                    case SELECTED_EFFECT_ID_AXIS:
                      {
                        selectedEffectId = SELECTED_EFFECT_ID_GLASSES;
                        break;
                      }

                    case SELECTED_EFFECT_ID_FACEPAINT:
                      {
                        selectedEffectId = SELECTED_EFFECT_ID_AXIS;
                        break;
                      }

                    case SELECTED_EFFECT_ID_GLASSES:
                      {
                        selectedEffectId = SELECTED_EFFECT_ID_FACEPAINT;
                        break;
                      }

                    default:
                      break;
                  }
                }
              }
            });
  }

  @Override
  public boolean onTouchEvent(MotionEvent event) {
    return tapGestureDetector.onTouchEvent(event);
  }

  private View createEffectSwitchingHintView() {
    TextView effectSwitchingHintView = new TextView(getApplicationContext());
    effectSwitchingHintView.setLayoutParams(
        new RelativeLayout.LayoutParams(LayoutParams.FILL_PARENT, LayoutParams.FILL_PARENT));
    effectSwitchingHintView.setText(EFFECT_SWITCHING_HINT_TEXT);
    effectSwitchingHintView.setGravity(Gravity.CENTER_HORIZONTAL | Gravity.BOTTOM);
    effectSwitchingHintView.setPadding(0, 0, 0, 480);
    effectSwitchingHintView.setTextColor(Color.parseColor("#ffffff"));
    effectSwitchingHintView.setTextSize((float) 24);

    return effectSwitchingHintView;
  }
}
