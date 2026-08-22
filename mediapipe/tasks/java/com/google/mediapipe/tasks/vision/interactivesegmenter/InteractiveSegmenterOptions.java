// Copyright 2026 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.vision.interactivesegmenter;

import com.google.auto.value.AutoValue;
import com.google.mediapipe.tasks.core.BaseOptions;

/** Options for setting up an {@link InteractiveSegmenter}. */
@AutoValue
public abstract class InteractiveSegmenterOptions {
  public abstract BaseOptions baseOptions();


  public static Builder builder() {
    return new AutoValue_InteractiveSegmenterOptions.Builder();
  }

  /** Builder for {@link InteractiveSegmenterOptions}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setBaseOptions(BaseOptions baseOptions);

    public abstract InteractiveSegmenterOptions build();
  }
}
