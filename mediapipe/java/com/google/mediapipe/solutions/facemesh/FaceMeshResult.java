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

package com.google.mediapipe.solutions.facemesh;

import android.graphics.Bitmap;
import com.google.auto.value.AutoBuilder;
import com.google.common.collect.ImmutableList;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.TextureFrame;
import com.google.mediapipe.solutioncore.ImageSolutionResult;
import java.util.List;

/**
 * FaceMeshResult contains the face landmarks on each detected face, and the input {@link Bitmap} or
 * {@link TextureFrame}. If not in static image mode, the timestamp field will be set to the
 * timestamp of the corresponding input image.
 */
public class FaceMeshResult extends ImageSolutionResult {
  private final ImmutableList<NormalizedLandmarkList> multiFaceLandmarks;

  FaceMeshResult(
      ImmutableList<NormalizedLandmarkList> multiFaceLandmarks,
      Packet imagePacket,
      long timestamp) {
    this.multiFaceLandmarks = multiFaceLandmarks;
    this.timestamp = timestamp;
    this.imagePacket = imagePacket;
  }

  // Collection of detected/tracked faces, where each face is represented as a list of 468 face
  // landmarks and each landmark is composed of x, y and z. x and y are normalized to [0.0, 1.0] by
  // the image width and height respectively. z represents the landmark depth with the depth at
  // center of the head being the origin, and the smaller the value the closer the landmark is to
  // the camera. The magnitude of z uses roughly the same scale as x.
  public ImmutableList<NormalizedLandmarkList> multiFaceLandmarks() {
    return multiFaceLandmarks;
  }

  public static Builder builder() {
    return new AutoBuilder_FaceMeshResult_Builder();
  }

  /** Builder for {@link FaceMeshResult}. */
  @AutoBuilder
  public abstract static class Builder {
    abstract Builder setMultiFaceLandmarks(List<NormalizedLandmarkList> value);

    abstract Builder setTimestamp(long value);

    abstract Builder setImagePacket(Packet value);

    abstract FaceMeshResult build();
  }
}
