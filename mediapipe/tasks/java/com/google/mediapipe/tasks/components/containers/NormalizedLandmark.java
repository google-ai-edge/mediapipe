// Copyright 2022 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.components.containers;

import android.annotation.TargetApi;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.formats.proto.LandmarkProto;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Normalized Landmark represents a point in 3D space with x, y, z coordinates. x and y are
 * normalized to [0.0, 1.0] by the image width and height respectively. z represents the landmark
 * depth, and the smaller the value the closer the landmark is to the camera. The magnitude of z
 * uses roughly the same scale as x.
 */
@AutoValue
@TargetApi(31)
public abstract class NormalizedLandmark {
  private static final float TOLERANCE = 1e-6f;

  /** Creates a normalized landmark from x, y, z coordinates. */
  public static NormalizedLandmark create(float x, float y, float z) {
    return new AutoValue_NormalizedLandmark(x, y, z, Optional.empty(), Optional.empty());
  }

  /**
   * Creates a normalized landmark from x, y, z coordinates with optional visibility and presence.
   */
  public static NormalizedLandmark create(
      float x, float y, float z, Optional<Float> visibility, Optional<Float> presence) {
    return new AutoValue_NormalizedLandmark(x, y, z, visibility, presence);
  }

  /** Creates a normalized landmark from a normalized landmark proto. */
  public static NormalizedLandmark createFromProto(LandmarkProto.NormalizedLandmark landmarkProto) {
    return NormalizedLandmark.create(
        landmarkProto.getX(),
        landmarkProto.getY(),
        landmarkProto.getZ(),
        landmarkProto.hasVisibility()
            ? Optional.of(landmarkProto.getVisibility())
            : Optional.empty(),
        landmarkProto.hasPresence() ? Optional.of(landmarkProto.getPresence()) : Optional.empty());
  }

  /** Creates a list of normalized landmarks from a {@link NormalizedLandmarkList}. */
  public static List<NormalizedLandmark> createListFromProto(
      LandmarkProto.NormalizedLandmarkList landmarkListProto) {
    List<NormalizedLandmark> landmarkList = new ArrayList<>();
    for (LandmarkProto.NormalizedLandmark landmarkProto : landmarkListProto.getLandmarkList()) {
      landmarkList.add(createFromProto(landmarkProto));
    }
    return landmarkList;
  }

  // The x coordinates of the normalized landmark.
  public abstract float x();

  // The y coordinates of the normalized landmark.
  public abstract float y();

  // The z coordinates of the normalized landmark.
  public abstract float z();

  // Visibility of the normalized landmark.
  public abstract Optional<Float> visibility();

  // Presence of the normalized landmark.
  public abstract Optional<Float> presence();

  @Override
  public final boolean equals(Object o) {
    if (!(o instanceof NormalizedLandmark)) {
      return false;
    }
    NormalizedLandmark other = (NormalizedLandmark) o;
    return Math.abs(other.x() - this.x()) < TOLERANCE
        && Math.abs(other.x() - this.y()) < TOLERANCE
        && Math.abs(other.x() - this.z()) < TOLERANCE;
  }

  @Override
  public final int hashCode() {
    return Objects.hash(x(), y(), z());
  }

  @Override
  public final String toString() {
    return "<Normalized Landmark (x="
        + x()
        + " y="
        + y()
        + " z="
        + z()
        + " visibility= "
        + visibility()
        + " presence="
        + presence()
        + ")>";
  }
}
