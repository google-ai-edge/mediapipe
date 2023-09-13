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

import android.graphics.RectF;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBox;
import com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypoint;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Represents one detected object in the results of {@link
 * com.google.mediapipe.tasks.version.objectdetector.ObjectDetector}.
 */
@AutoValue
public abstract class Detection {

  private static final int DEFAULT_CATEGORY_INDEX = -1;

  /**
   * Creates a {@link Detection} instance from a list of {@link Category} and a bounding box.
   *
   * @param categories a list of {@link Category} objects that contain category name, display name,
   *     score, and the label index.
   * @param boundingBox a {@link RectF} object to represent the bounding box.
   */
  public static Detection create(List<Category> categories, RectF boundingBox) {

    // As an open source project, we've been trying avoiding depending on common java libraries,
    // such as Guava, because it may introduce conflicts with clients who also happen to use those
    // libraries. Therefore, instead of using ImmutableList here, we convert the List into
    // unmodifiableList
    return new AutoValue_Detection(
        Collections.unmodifiableList(categories), boundingBox, Optional.empty());
  }

  /**
   * Creates a {@link Detection} instance from a list of {@link Category} and a bounding box.
   *
   * @param categories a list of {@link Category} objects that contain category name, display name,
   *     score, and the label index.
   * @param boundingBox a {@link RectF} object to represent the bounding box.
   * @param keypoints an optional list of {@link NormalizedKeypoints} associated with the detection.
   */
  public static Detection create(
      List<Category> categories, RectF boundingBox, Optional<List<NormalizedKeypoint>> keypoints) {
    return new AutoValue_Detection(
        Collections.unmodifiableList(categories), boundingBox, keypoints);
  }

  /**
   * Creates a {@link Detection} instance from a {@link
   * com.google.mediapipe.formats.proto.DetectionProto.Detection} protobuf message.
   *
   * @param detectionProto a {@link com.google.mediapipe.formats.proto.DetectionProto.Detection}
   *     protobuf message.
   */
  public static Detection createFromProto(
      com.google.mediapipe.formats.proto.DetectionProto.Detection detectionProto) {
    List<Category> categories = new ArrayList<>();
    for (int idx = 0; idx < detectionProto.getScoreCount(); ++idx) {
      categories.add(
          Category.create(
              detectionProto.getScore(idx),
              detectionProto.getLabelIdCount() > idx
                  ? detectionProto.getLabelId(idx)
                  : DEFAULT_CATEGORY_INDEX,
              detectionProto.getLabelCount() > idx ? detectionProto.getLabel(idx) : "",
              detectionProto.getDisplayNameCount() > idx
                  ? detectionProto.getDisplayName(idx)
                  : ""));
    }
    RectF boundingBox = new RectF();
    if (detectionProto.getLocationData().hasBoundingBox()) {
      BoundingBox boundingBoxProto = detectionProto.getLocationData().getBoundingBox();
      boundingBox.set(
          /* left= */ boundingBoxProto.getXmin(),
          /* top= */ boundingBoxProto.getYmin(),
          /* right= */ boundingBoxProto.getXmin() + boundingBoxProto.getWidth(),
          /* bottom= */ boundingBoxProto.getYmin() + boundingBoxProto.getHeight());
    }
    Optional<List<NormalizedKeypoint>> keypoints = Optional.empty();
    if (!detectionProto.getLocationData().getRelativeKeypointsList().isEmpty()) {
      keypoints = Optional.of(new ArrayList<>());
      for (RelativeKeypoint relativeKeypoint :
          detectionProto.getLocationData().getRelativeKeypointsList()) {
        keypoints
            .get()
            .add(
                NormalizedKeypoint.create(
                    relativeKeypoint.getX(),
                    relativeKeypoint.getY(),
                    relativeKeypoint.hasKeypointLabel()
                        ? Optional.of(relativeKeypoint.getKeypointLabel())
                        : Optional.empty(),
                    relativeKeypoint.hasScore()
                        ? Optional.of(relativeKeypoint.getScore())
                        : Optional.empty()));
      }
    }
    return create(categories, boundingBox, keypoints);
  }

  /** A list of {@link Category} objects. */
  public abstract List<Category> categories();

  /** A {@link RectF} object to represent the bounding box of the detected object. */
  public abstract RectF boundingBox();

  /**
   * An optional list of {@link NormalizedKeypoint} associated with the detection. Keypoints
   * represent interesting points related to the detection. For example, the keypoints represent the
   * eyes, ear and mouth from face detection model. Or in the template matching detection, e.g.
   * KNIFT, they can represent the feature points for template matching.
   */
  public abstract Optional<List<NormalizedKeypoint>> keypoints();
}
