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

package ca.copperlabs.mediapipe.posetracking;

import com.google.auto.value.AutoValue;

import ca.copperlabs.mediapipe.posetracking.AutoValue_PoseTrackingOptions;

/**
 * MediaPipe Face Detection solution-specific options.
 *
 * <p>staticImageMode: Whether to treat the input images as a batch of static and possibly unrelated
 * images, or a video stream. Default to false. See details in
 * https://solutions.mediapipe.dev/face_detection#static_image_mode.
 *
 * <p>minDetectionConfidence: Minimum confidence value ([0.0, 1.0]) for face detection to be
 * considered successful. See details in
 * https://solutions.mediapipe.dev/face_detection#min_detection_confidence.
 *
 * <p>modelSelection: 0 or 1. 0 to select a short-range model that works best for faces within 2
 * meters from the camera, and 1 for a full-range model best for faces within 5 meters. See details
 * in https://solutions.mediapipe.dev/face_detection#model_selection.
 */
@AutoValue
public abstract class PoseTrackingOptions {
  public abstract boolean staticImageMode();

  public abstract int modelComplexity();

  public abstract float minDetectionConfidence();

  public abstract boolean landmarkVisibility();

  public abstract boolean smoothLandmarks();

  public static Builder builder() {
    return new AutoValue_PoseTrackingOptions.Builder().withDefaultValues();
  }

  /** Builder for {@link PoseTrackingOptions}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public Builder withDefaultValues() {
      return setStaticImageMode(false)
              .setModelComplexity(0)
              .setMinDetectionConfidence(0.5f)
              .setSmoothLandmarks(true);
    }
    public Builder withPoseTrackingOptions(PoseTrackingOptions options){
      return setStaticImageMode(options.staticImageMode())
              .setModelComplexity(options.modelComplexity())
              .setMinDetectionConfidence(options.minDetectionConfidence())
              .setSmoothLandmarks(options.smoothLandmarks())
              .setLandmarkVisibility(options.landmarkVisibility());
    }

    public abstract Builder setStaticImageMode(boolean value);

    public abstract Builder setModelComplexity(int value);

    public abstract Builder setMinDetectionConfidence(float value);

    public abstract Builder setLandmarkVisibility(boolean value);

    public abstract Builder setSmoothLandmarks(boolean value);
    public abstract PoseTrackingOptions build();
  }
}
