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

package com.google.mediapipe.tasks.components.processors;

import com.google.auto.value.AutoValue;
import com.google.mediapipe.tasks.components.processors.proto.ClassifierOptionsProto;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/** Classifier options shared across MediaPipe Java classification tasks. */
@AutoValue
public abstract class ClassifierOptions {

  /** Builder for {@link ClassifierOptions}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /**
     * Sets the optional locale to use for display names specified through the TFLite Model
     * Metadata, if any.
     */
    public abstract Builder setDisplayNamesLocale(String locale);

    /**
     * Sets the optional maximum number of top-scored classification results to return.
     *
     * <p>If not set, all available results are returned. If set, must be > 0.
     */
    public abstract Builder setMaxResults(Integer maxResults);

    /**
     * Sets the optional score threshold. Results with score below this value are rejected.
     *
     * <p>Overrides the score threshold specified in the TFLite Model Metadata, if any.
     */
    public abstract Builder setScoreThreshold(Float scoreThreshold);

    /**
     * Sets the optional allowlist of category names.
     *
     * <p>If non-empty, detection results whose category name is not in this set will be filtered
     * out. Duplicate or unknown category names are ignored. Mutually exclusive with {@code
     * categoryDenylist}.
     */
    public abstract Builder setCategoryAllowlist(List<String> categoryAllowlist);

    /**
     * Sets the optional denylist of category names.
     *
     * <p>If non-empty, detection results whose category name is in this set will be filtered out.
     * Duplicate or unknown category names are ignored. Mutually exclusive with {@code
     * categoryAllowlist}.
     */
    public abstract Builder setCategoryDenylist(List<String> categoryDenylist);

    abstract ClassifierOptions autoBuild();

    /**
     * Validates and builds the {@link ClassifierOptions} instance.
     *
     * @throws IllegalArgumentException if {@link maxResults} is set to a value <= 0.
     */
    public final ClassifierOptions build() {
      ClassifierOptions options = autoBuild();
      if (options.maxResults().isPresent() && options.maxResults().get() <= 0) {
        throw new IllegalArgumentException("If specified, maxResults must be > 0");
      }
      return options;
    }
  }

  public abstract Optional<String> displayNamesLocale();

  public abstract Optional<Integer> maxResults();

  public abstract Optional<Float> scoreThreshold();

  public abstract List<String> categoryAllowlist();

  public abstract List<String> categoryDenylist();

  public static Builder builder() {
    return new AutoValue_ClassifierOptions.Builder()
        .setCategoryAllowlist(Collections.emptyList())
        .setCategoryDenylist(Collections.emptyList());
  }

  /**
   * Converts a {@link ClassifierOptions} object to a {@link
   * ClassifierOptionsProto.ClassifierOptions} protobuf message.
   */
  public ClassifierOptionsProto.ClassifierOptions convertToProto() {
    ClassifierOptionsProto.ClassifierOptions.Builder builder =
        ClassifierOptionsProto.ClassifierOptions.newBuilder();
    displayNamesLocale().ifPresent(builder::setDisplayNamesLocale);
    maxResults().ifPresent(builder::setMaxResults);
    scoreThreshold().ifPresent(builder::setScoreThreshold);
    if (!categoryAllowlist().isEmpty()) {
      builder.addAllCategoryAllowlist(categoryAllowlist());
    }
    if (!categoryDenylist().isEmpty()) {
      builder.addAllCategoryDenylist(categoryDenylist());
    }
    return builder.build();
  }
}
