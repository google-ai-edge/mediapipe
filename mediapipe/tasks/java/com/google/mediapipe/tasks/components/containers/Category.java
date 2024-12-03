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

import com.google.auto.value.AutoValue;
import com.google.mediapipe.formats.proto.ClassificationProto;
import com.google.mediapipe.formats.proto.ClassificationProto.Classification;
import com.google.mediapipe.formats.proto.ClassificationProto.ClassificationList;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Category is a util class, that contains a category name, its display name, a float value as
 * score, and the index of the label in the corresponding label file. Typically it's used as result
 * of classification or detection tasks.
 */
@AutoValue
public abstract class Category {
  private static final float TOLERANCE = 1e-6f;

  /**
   * Creates a {@link Category} instance.
   *
   * @param score the probability score of this label category.
   * @param index the index of the label in the corresponding label file.
   * @param categoryName the label of this category object.
   * @param displayName the display name of the label.
   */
  public static Category create(float score, int index, String categoryName, String displayName) {
    return new AutoValue_Category(score, index, categoryName, displayName);
  }

  /**
   * Creates a {@link Category} object from a {@link ClassificationProto.Classification} protobuf
   * message.
   *
   * @param proto the {@link ClassificationProto.Classification} protobuf message to convert.
   */
  public static Category createFromProto(ClassificationProto.Classification proto) {
    return create(proto.getScore(), proto.getIndex(), proto.getLabel(), proto.getDisplayName());
  }

  /**
   * Creates a list of {@link Category} objects from a {@link
   * ClassificationProto.ClassificationList}.
   *
   * @param classificationListProto the {@link ClassificationProto.ClassificationList} protobuf
   *     message to convert.
   * @return A list of {@link Category} objects.
   */
  public static List<Category> createListFromProto(ClassificationList classificationListProto) {
    List<Category> categoryList = new ArrayList<>();
    for (Classification classification : classificationListProto.getClassificationList()) {
      categoryList.add(createFromProto(classification));
    }
    return categoryList;
  }

  /** The probability score of this label category. */
  public abstract float score();

  /** The index of the label in the corresponding label file. Returns -1 if the index is not set. */
  public abstract int index();

  /** The label of this category object. */
  public abstract String categoryName();

  /**
   * The display name of the label, which may be translated for different locales. For example, a
   * label, "apple", may be translated into Spanish for display purpose, so that the display name is
   * "manzana".
   */
  public abstract String displayName();

  @Override
  public final boolean equals(Object o) {
    if (!(o instanceof Category)) {
      return false;
    }
    Category other = (Category) o;
    return Math.abs(other.score() - this.score()) < TOLERANCE
        && other.index() == this.index()
        && other.categoryName().equals(this.categoryName())
        && other.displayName().equals(this.displayName());
  }

  @Override
  public final int hashCode() {
    return Objects.hash(categoryName(), displayName(), score(), index());
  }

  @Override
  public final String toString() {
    return "<Category \""
        + categoryName()
        + "\" (displayName="
        + displayName()
        + " score="
        + score()
        + " index="
        + index()
        + ")>";
  }
}
