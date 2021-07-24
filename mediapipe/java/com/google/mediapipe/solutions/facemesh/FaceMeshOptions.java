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

import androidx.annotation.IntDef;
import com.google.auto.value.AutoValue;

/**
 * MediaPipe FaceMesh solution-specific options.
 *
 * <p>mode: Whether to treat the input images as a batch of static and possibly unrelated images, or
 * a video stream. See details in https://solutions.mediapipe.dev/face_mesh#static_image_mode.
 *
 * <p>maxNumFaces: Maximum number of faces to detect. See details in
 * https://solutions.mediapipe.dev/face_mesh#max_num_faces.
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

  // TODO: Switch to use boolean variable.
  public static final int STREAMING_MODE = 1;
  public static final int STATIC_IMAGE_MODE = 2;

  /**
   * Indicates whether to treat the input images as a batch of static and possibly unrelated images,
   * or a video stream.
   */
  @IntDef({STREAMING_MODE, STATIC_IMAGE_MODE})
  public @interface Mode {}

  @Mode
  public abstract int mode();

  public abstract int maxNumFaces();

  public abstract float minDetectionConfidence();

  public abstract float minTrackingConfidence();

  public abstract boolean runOnGpu();

  public static Builder builder() {
    return new AutoValue_FaceMeshOptions.Builder().withDefaultValues();
  }

  /** Builder for {@link FaceMeshOptions}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public Builder withDefaultValues() {
      return setMaxNumFaces(1)
          .setMinDetectionConfidence(0.5f)
          .setMinTrackingConfidence(0.5f)
          .setRunOnGpu(true);
    }

    public abstract Builder setMode(int value);

    public abstract Builder setMaxNumFaces(int value);

    public abstract Builder setMinDetectionConfidence(float value);

    public abstract Builder setMinTrackingConfidence(float value);

    public abstract Builder setRunOnGpu(boolean value);

    public abstract FaceMeshOptions build();
  }
}
