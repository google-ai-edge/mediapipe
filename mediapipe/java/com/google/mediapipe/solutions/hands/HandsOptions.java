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
import com.google.auto.value.AutoValue;

/**
 * MediaPipe Hands solution-specific options.
 *
 * <p>mode: Whether to treat the input images as a batch of static and possibly unrelated images, or
 * a video stream. See details in https://solutions.mediapipe.dev/hands#static_image_mode.
 *
 * <p>maxNumHands: Maximum number of hands to detect. See details in
 * https://solutions.mediapipe.dev/hands#max_num_hands.
 *
 * <p>minDetectionConfidence: Minimum confidence value ([0.0, 1.0]) for hand detection to be
 * considered successful. See details in
 * https://solutions.mediapipe.dev/hands#min_detection_confidence.
 *
 * <p>minTrackingConfidence: Minimum confidence value ([0.0, 1.0]) for the hand landmarks to be
 * considered tracked successfully. See details in
 * https://solutions.mediapipe.dev/hands#min_tracking_confidence.
 *
 * <p>runOnGpu: Whether to run pipeline on GPU or CPU. Default to true.
 */
@AutoValue
public abstract class HandsOptions {

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

  public abstract int maxNumHands();

  public abstract float minDetectionConfidence();

  public abstract float minTrackingConfidence();

  public abstract boolean runOnGpu();

  public static Builder builder() {
    return new AutoValue_HandsOptions.Builder().withDefaultValues();
  }

  /** Builder for {@link HandsOptions}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public Builder withDefaultValues() {
      return setMaxNumHands(2)
          .setMinDetectionConfidence(0.5f)
          .setMinTrackingConfidence(0.5f)
          .setRunOnGpu(true);
    }

    public abstract Builder setMode(int value);

    public abstract Builder setMaxNumHands(int value);

    public abstract Builder setMinDetectionConfidence(float value);

    public abstract Builder setMinTrackingConfidence(float value);

    public abstract Builder setRunOnGpu(boolean value);

    public abstract HandsOptions build();
  }
}
