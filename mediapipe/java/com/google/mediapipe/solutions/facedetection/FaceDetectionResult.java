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

import android.graphics.Bitmap;
import com.google.auto.value.AutoBuilder;
import com.google.common.collect.ImmutableList;
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
public class FaceDetectionResult extends ImageSolutionResult {
  private final ImmutableList<Detection> multiFaceDetections;

  FaceDetectionResult(
      ImmutableList<Detection> multiFaceDetections, Packet imagePacket, long timestamp) {
    this.multiFaceDetections = multiFaceDetections;
    this.timestamp = timestamp;
    this.imagePacket = imagePacket;
  }

  // Collection of detected faces, where each face is represented as a detection proto message that
  // contains a bounding box and 6 {@link FaceKeypoint}s. The bounding box is composed of xmin and
  // width (both normalized to [0.0, 1.0] by the image width) and ymin and height (both normalized
  // to [0.0, 1.0] by the image height). Each keypoint is composed of x and y, which are normalized
  // to [0.0, 1.0] by the image width and height respectively.
  public ImmutableList<Detection> multiFaceDetections() {
    return multiFaceDetections;
  }

  public static Builder builder() {
    return new AutoBuilder_FaceDetectionResult_Builder();
  }

  /** Builder for {@link FaceDetectionResult}. */
  @AutoBuilder
  public abstract static class Builder {
    abstract Builder setMultiFaceDetections(List<Detection> value);

    abstract Builder setTimestamp(long value);

    abstract Builder setImagePacket(Packet value);

    abstract FaceDetectionResult build();
  }
}
