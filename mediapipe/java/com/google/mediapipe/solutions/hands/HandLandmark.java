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

package com.google.mediapipe.solutions.hands;

import androidx.annotation.IntDef;

/** The 21 hand landmarks. */
public final class HandLandmark {
  public static final int NUM_LANDMARKS = 21;

  public static final int WRIST = 0;
  public static final int THUMB_CMC = 1;
  public static final int THUMB_MCP = 2;
  public static final int THUMB_IP = 3;
  public static final int THUMB_TIP = 4;
  public static final int INDEX_FINGER_MCP = 5;
  public static final int INDEX_FINGER_PIP = 6;
  public static final int INDEX_FINGER_DIP = 7;
  public static final int INDEX_FINGER_TIP = 8;
  public static final int MIDDLE_FINGER_MCP = 9;
  public static final int MIDDLE_FINGER_PIP = 10;
  public static final int MIDDLE_FINGER_DIP = 11;
  public static final int MIDDLE_FINGER_TIP = 12;
  public static final int RING_FINGER_MCP = 13;
  public static final int RING_FINGER_PIP = 14;
  public static final int RING_FINGER_DIP = 15;
  public static final int RING_FINGER_TIP = 16;
  public static final int PINKY_MCP = 17;
  public static final int PINKY_PIP = 18;
  public static final int PINKY_DIP = 19;
  public static final int PINKY_TIP = 20;

  /** Represents a hand landmark type. */
  @IntDef({
    WRIST,
    THUMB_CMC,
    THUMB_MCP,
    THUMB_IP,
    THUMB_TIP,
    INDEX_FINGER_MCP,
    INDEX_FINGER_PIP,
    INDEX_FINGER_DIP,
    INDEX_FINGER_TIP,
    MIDDLE_FINGER_MCP,
    MIDDLE_FINGER_PIP,
    MIDDLE_FINGER_DIP,
    MIDDLE_FINGER_TIP,
    RING_FINGER_MCP,
    RING_FINGER_PIP,
    RING_FINGER_DIP,
    RING_FINGER_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_DIP,
    PINKY_TIP,
  })
  public @interface HandLandmarkType {}

  private HandLandmark() {}
}
