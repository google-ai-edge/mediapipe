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

import com.google.auto.value.AutoValue;

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
public abstract class FaceDetectionOptions {
  public abstract boolean staticImageMode();

  public abstract int modelSelection();

  public abstract float minDetectionConfidence();

  public static Builder builder() {
    return new AutoValue_FaceDetectionOptions.Builder().withDefaultValues();
  }

  /** Builder for {@link FaceDetectionOptions}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public Builder withDefaultValues() {
      return setStaticImageMode(false).setModelSelection(0).setMinDetectionConfidence(0.5f);
    }

    public abstract Builder setStaticImageMode(boolean value);

    public abstract Builder setModelSelection(int value);

    public abstract Builder setMinDetectionConfidence(float value);

    public abstract FaceDetectionOptions build();
  }
}
