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

import android.graphics.Bitmap;
import com.google.auto.value.AutoBuilder;
import com.google.common.collect.ImmutableList;
import com.google.mediapipe.formats.proto.LandmarkProto.LandmarkList;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.formats.proto.ClassificationProto.Classification;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.TextureFrame;
import com.google.mediapipe.solutioncore.ImageSolutionResult;
import java.util.List;

/**
 * HandsResult contains a collection of detected/tracked hands, a collection of handedness of the
 * detected/tracked hands, and the input {@link Bitmap} or {@link TextureFrame}. If not in static
 * image mode, the timestamp field will be set to the timestamp of the corresponding input image.
 */
public class HandsResult extends ImageSolutionResult {
  private final ImmutableList<NormalizedLandmarkList> multiHandLandmarks;
  private final ImmutableList<LandmarkList> multiHandWorldLandmarks;
  private final ImmutableList<Classification> multiHandedness;

  HandsResult(
      ImmutableList<NormalizedLandmarkList> multiHandLandmarks,
      ImmutableList<LandmarkList> multiHandWorldLandmarks,
      ImmutableList<Classification> multiHandedness,
      Packet imagePacket,
      long timestamp) {
    this.multiHandLandmarks = multiHandLandmarks;
    this.multiHandWorldLandmarks = multiHandWorldLandmarks;
    this.multiHandedness = multiHandedness;
    this.timestamp = timestamp;
    this.imagePacket = imagePacket;
  }

  // Collection of detected/tracked hands, where each hand is represented as a list of 21 hand
  // landmarks and each landmark is composed of x, y and z. x and y are normalized to [0.0, 1.0] by
  // the image width and height respectively. z represents the landmark depth with the depth at the
  // wrist being the origin, and the smaller the value the closer the landmark is to the camera. The
  // magnitude of z uses roughly the same scale as x.
  public ImmutableList<NormalizedLandmarkList> multiHandLandmarks() {
    return multiHandLandmarks;
  }

  // Collection of detected/tracked hands' landmarks in real-world 3D coordinates that are in meters
  // with the origin at the hand's approximate geometric center.
  public ImmutableList<LandmarkList> multiHandWorldLandmarks() {
    return multiHandWorldLandmarks;
  }

  // Collection of handedness of the detected/tracked hands (i.e. is it a left or right hand). Each
  // hand is composed of label and score. label is a string of value either "Left" or "Right". score
  // is the estimated probability of the predicted handedness and is always greater than or equal to
  // 0.5 (and the opposite handedness has an estimated probability of 1 - score).
  public ImmutableList<Classification> multiHandedness() {
    return multiHandedness;
  }

  public static Builder builder() {
    return new AutoBuilder_HandsResult_Builder();
  }

  /** Builder for {@link HandsResult}. */
  @AutoBuilder
  public abstract static class Builder {
    abstract Builder setMultiHandLandmarks(List<NormalizedLandmarkList> value);

    abstract Builder setMultiHandWorldLandmarks(List<LandmarkList> value);

    abstract Builder setMultiHandedness(List<Classification> value);

    abstract Builder setTimestamp(long value);

    abstract Builder setImagePacket(Packet value);

    abstract HandsResult build();
  }
}
