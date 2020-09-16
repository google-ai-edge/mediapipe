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
import java.util.List;

/** Main activity of MediaPipe face mesh app. */
public class MainActivity extends com.google.mediapipe.apps.basic.MainActivity {
  private static final String TAG = "MainActivity";

  // Stream names.
  private static final String IS_FACEPAINT_EFFECT_SELECTED_INPUT_STREAM_NAME =
      "is_facepaint_effect_selected";
  private static final String OUTPUT_FACE_GEOMETRY_STREAM_NAME = "multi_face_geometry";

  private static final String EFFECT_SWITCHING_HINT_TEXT = "Tap to switch between effects!";

  private static final int MATRIX_TRANSLATION_Z_INDEX = 14;

  private final Object isFacepaintEffectSelectedLock = new Object();
  private boolean isFacepaintEffectSelected;

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

    // By default, render the glasses effect.
    isFacepaintEffectSelected = false;

    // This callback demonstrates how the output face geometry packet can be obtained and used
    // in an Android app. As an example, the Z-translation component of the face pose transform
    // matrix is logged for each face being equal to the approximate distance away from the camera
    // in centimeters.
    processor.addPacketCallback(
        OUTPUT_FACE_GEOMETRY_STREAM_NAME,
        (packet) -> {
          effectSwitchingHintView.post(
              new Runnable() {
                @Override
                public void run() {
                  effectSwitchingHintView.setVisibility(View.VISIBLE);
                }
              });

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

    // Alongside the input camera frame, we also send the `is_facepaint_effect_selected` boolean
    // packet to indicate which effect should be rendered on this frame.
    processor.setOnWillAddFrameListener(
        (timestamp) -> {
          Packet isFacepaintEffectSelectedPacket = null;
          try {
            synchronized (isFacepaintEffectSelectedLock) {
              isFacepaintEffectSelectedPacket =
                  processor.getPacketCreator().createBool(isFacepaintEffectSelected);
            }

            processor
                .getGraph()
                .addPacketToInputStream(
                    IS_FACEPAINT_EFFECT_SELECTED_INPUT_STREAM_NAME,
                    isFacepaintEffectSelectedPacket,
                    timestamp);
          } catch (RuntimeException e) {
            Log.e(
                TAG,
                "Exception while adding packet to input stream while switching effects: " + e);
          } finally {
            if (isFacepaintEffectSelectedPacket != null) {
              isFacepaintEffectSelectedPacket.release();
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
                synchronized (isFacepaintEffectSelectedLock) {
                  isFacepaintEffectSelected = !isFacepaintEffectSelected;
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
