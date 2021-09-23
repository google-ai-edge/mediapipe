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

package com.google.mediapipe.solutioncore;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;

/** SolutionInfo contains all needed informaton to initialize a MediaPipe solution graph. */
@AutoValue
public abstract class SolutionInfo {
  public abstract String binaryGraphPath();

  public abstract String imageInputStreamName();

  public abstract ImmutableList<String> outputStreamNames();

  public abstract boolean staticImageMode();

  public static Builder builder() {
    return new AutoValue_SolutionInfo.Builder();
  }

  /** Builder for {@link SolutionInfo}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setBinaryGraphPath(String value);

    public abstract Builder setImageInputStreamName(String value);

    public abstract Builder setOutputStreamNames(ImmutableList<String> value);

    public abstract Builder setStaticImageMode(boolean value);

    public abstract SolutionInfo build();
  }
}
