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

package com.google.mediapipe.solutions.posetracking;

import android.graphics.Bitmap;
import com.google.auto.value.AutoBuilder;
import com.google.common.collect.ImmutableList;
import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.TextureFrame;
import com.google.mediapipe.solutioncore.ImageSolutionResult;
import com.google.mediapipe.formats.proto.DetectionProto.Detection;
import java.util.List;



/**
 * FaceDetectionResult contains the detected faces, and the input {@link Bitmap} or {@link
 * TextureFrame}. If not in static image mode, the timestamp field will be set to the timestamp of
 * the corresponding input image.
 */
public class PoseTrackingResult extends ImageSolutionResult {
  private final ImmutableList<Detection> multiPoseDetections;
  private final ImmutableList<LandmarkProto.Landmark> multiPoseLandmarks;

  public static final int NOSE = 0;
  public static final int LEFT_EYE_INNER = 1;
  public static final int LEFT_EYE = 2;
  public static final int LEFT_EYE_OUTER = 3;
  public static final int RIGHT_EYE_INNER = 4;
  public static final int RIGHT_EYE = 5;
  public static final int RIGHT_EYE_OUTER = 6;
  public static final int LEFT_EAR = 7;
  public static final int RIGHT_EAR = 8;
  public static final int MOUTH_LEFT = 9;
  public static final int MOUTH_RIGHT = 10;
  public static final int LEFT_SHOULDER = 11;
  public static final int RIGHT_SHOULDER = 12;
  public static final int LEFT_ELBOW = 13;
  public static final int RIGHT_ELBOW = 14;
  public static final int LEFT_WRIST = 15;
  public static final int RIGHT_WRIST = 16;
  public static final int LEFT_PINKY = 17;
  public static final int RIGHT_PINKY = 18;
  public static final int LEFT_INDEX = 19;
  public static final int RIGHT_INDEX = 20;
  public static final int LEFT_THUMB = 21;
  public static final int RIGHT_THUMB = 22;
  public static final int LEFT_HIP = 23;
  public static final int RIGHT_HIP = 24;
  public static final int LEFT_KNEE = 25;
  public static final int RIGHT_KNEE = 26;
  public static final int LEFT_ANKLE = 27;
  public static final int RIGHT_ANKLE = 28;
  public static final int LEFT_HEEL = 29;
  public static final int RIGHT_HEEL = 30;
  public static final int LEFT_FOOT = 31;
  public static final int RIGHT_FOOT = 32;


  PoseTrackingResult(
          ImmutableList<Detection> multiPoseDetections,ImmutableList<LandmarkProto.Landmark> multiPoseLandmarks, Packet imagePacket, long timestamp) {
    this.multiPoseDetections = multiPoseDetections;
    this.multiPoseLandmarks = multiPoseLandmarks;
    this.timestamp = timestamp;
    this.imagePacket = imagePacket;
  }

  // Collection of detected faces, where each face is represented as a detection proto message that
  // contains a bounding box and 6 {@link FaceKeypoint}s. The bounding box is composed of xmin and
  // width (both normalized to [0.0, 1.0] by the image width) and ymin and height (both normalized
  // to [0.0, 1.0] by the image height). Each keypoint is composed of x and y, which are normalized
  // to [0.0, 1.0] by the image width and height respectively.
  public ImmutableList<Detection> multiPoseTrackings() {
    return multiPoseDetections;
  }


  public ImmutableList<LandmarkProto.Landmark> multiPoseLandmarks() {
    return multiPoseLandmarks;
  }

  public static Builder builder() {
    return new AutoBuilder_PoseTrackingResult_Builder();
  }

  /** Builder for {@link PoseTrackingResult}. */
  @AutoBuilder
  public abstract static class Builder {
    abstract Builder setMultiPoseDetections(List<Detection> value);
    abstract Builder setMultiPoseLandmarks(List<LandmarkProto.Landmark> value);

    abstract Builder setTimestamp(long value);

    abstract Builder setImagePacket(Packet value);

    abstract PoseTrackingResult build();
  }
}
