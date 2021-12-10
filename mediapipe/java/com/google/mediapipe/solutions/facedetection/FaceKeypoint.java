// Copyright 2021 The MediaPipe Authors.
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

package com.google.mediapipe.solutions.facedetection;

import androidx.annotation.IntDef;

/** The 6 face keypoints. */
public final class FaceKeypoint {
  public static final int NUM_KEY_POINTS = 6;

  public static final int RIGHT_EYE = 0;
  public static final int LEFT_EYE = 1;
  public static final int NOSE_TIP = 2;
  public static final int MOUTH_CENTER = 3;
  public static final int RIGHT_EAR_TRAGION = 4;
  public static final int LEFT_EAR_TRAGION = 5;

  /** Represents a face keypoint type. */
  @IntDef({
    RIGHT_EYE,
    LEFT_EYE,
    NOSE_TIP,
    MOUTH_CENTER,
    RIGHT_EAR_TRAGION,
    LEFT_EAR_TRAGION,
  })
  public @interface FaceKeypointType {}

  private FaceKeypoint() {}
}
