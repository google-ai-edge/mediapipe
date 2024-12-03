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
 * Landmark represents a point in 3D space with x, y, z coordinates. The landmark coordinates are in
 * meters. z represents the landmark depth, and the smaller the value the closer the world landmark
 * is to the camera.
 */
@AutoValue
@TargetApi(31)
public abstract class Landmark {
  private static final float TOLERANCE = 1e-6f;

  /** Creates a landmark from x, y, z coordinates. */
  public static Landmark create(float x, float y, float z) {
    return new AutoValue_Landmark(x, y, z, Optional.empty(), Optional.empty());
  }

  /**
   * Creates a normalized landmark from x, y, z coordinates with optional visibility and presence.
   */
  public static Landmark create(
      float x, float y, float z, Optional<Float> visibility, Optional<Float> presence) {
    return new AutoValue_Landmark(x, y, z, visibility, presence);
  }

  /** Creates a landmark from a landmark proto. */
  public static Landmark createFromProto(LandmarkProto.Landmark landmarkProto) {
    return Landmark.create(
        landmarkProto.getX(),
        landmarkProto.getY(),
        landmarkProto.getZ(),
        landmarkProto.hasVisibility()
            ? Optional.of(landmarkProto.getVisibility())
            : Optional.empty(),
        landmarkProto.hasPresence() ? Optional.of(landmarkProto.getPresence()) : Optional.empty());
  }

  /** Creates a list of landmarks from a {@link LandmarkList}. */
  public static List<Landmark> createListFromProto(LandmarkProto.LandmarkList landmarkListProto) {
    List<Landmark> landmarkList = new ArrayList<>();
    for (LandmarkProto.Landmark landmarkProto : landmarkListProto.getLandmarkList()) {
      landmarkList.add(createFromProto(landmarkProto));
    }
    return landmarkList;
  }

  // The x coordinates of the landmark.
  public abstract float x();

  // The y coordinates of the landmark.
  public abstract float y();

  // The z coordinates of the landmark.
  public abstract float z();

  // Visibility of the normalized landmark.
  public abstract Optional<Float> visibility();

  // Presence of the normalized landmark.
  public abstract Optional<Float> presence();

  @Override
  public final boolean equals(Object o) {
    if (!(o instanceof Landmark)) {
      return false;
    }
    Landmark other = (Landmark) o;
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
    return "<Landmark (x="
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
