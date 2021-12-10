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

import com.google.auto.value.AutoValue;

/**
 * MediaPipe FaceMesh solution-specific options.
 *
 * <p>staticImageMode: Whether to treat the input images as a batch of static and possibly unrelated
 * images, or a video stream. Default to false. See details in
 * https://solutions.mediapipe.dev/face_mesh#static_image_mode.
 *
 * <p>maxNumFaces: Maximum number of faces to detect. See details in
 * https://solutions.mediapipe.dev/face_mesh#max_num_faces.
 *
 * <p>refineLandmarks: Whether to further refine the landmark coordinates around the eyes, lips and
 * face oval, and output additional landmarks around the irises. Default to False. See details in
 * https://solutions.mediapipe.dev/face_mesh#refine_landmark.
 *
 * <p>minDetectionConfidence: Minimum confidence value ([0.0, 1.0]) for face detection to be
 * considered successful. See details in
 * https://solutions.mediapipe.dev/face_mesh#min_detection_confidence.
 *
 * <p>minTrackingConfidence: Minimum confidence value ([0.0, 1.0]) for the face landmarks to be
 * considered tracked successfully. See details in
 * https://solutions.mediapipe.dev/face_mesh#min_tracking_confidence.
 *
 * <p>runOnGpu: Whether to run pipeline on GPU or CPU. Default to true.
 */
@AutoValue
public abstract class FaceMeshOptions {

  public abstract boolean staticImageMode();

  public abstract int maxNumFaces();

  public abstract float minDetectionConfidence();

  public abstract float minTrackingConfidence();

  public abstract boolean refineLandmarks();

  public abstract boolean runOnGpu();

  public static Builder builder() {
    return new AutoValue_FaceMeshOptions.Builder().withDefaultValues();
  }

  /** Builder for {@link FaceMeshOptions}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public Builder withDefaultValues() {
      return setStaticImageMode(false)
          .setMaxNumFaces(1)
          .setMinDetectionConfidence(0.5f)
          .setMinTrackingConfidence(0.5f)
          .setRefineLandmarks(false)
          .setRunOnGpu(true);
    }

    public abstract Builder setStaticImageMode(boolean value);

    public abstract Builder setMaxNumFaces(int value);

    public abstract Builder setMinDetectionConfidence(float value);

    public abstract Builder setMinTrackingConfidence(float value);

    public abstract Builder setRefineLandmarks(boolean value);

    public abstract Builder setRunOnGpu(boolean value);

    public abstract FaceMeshOptions build();
  }
}
