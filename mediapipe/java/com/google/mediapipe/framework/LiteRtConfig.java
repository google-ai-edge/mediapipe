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

package com.google.mediapipe.framework;

import androidx.annotation.Nullable;
import com.google.auto.value.AutoValue;

/** Configuration for LiteRtService. */
@AutoValue
public abstract class LiteRtConfig {
  @Nullable
  public abstract String getDispatchLibraryPath();

  /**
   * Returns the handle object acquired through the Google Play Services API.
   *
   * <p>NOTE: This API only applies to Android.
   */
  @Nullable
  public abstract Object getSystemRuntimeHandle();

  public static Builder builder() {
    return new AutoValue_LiteRtConfig.Builder();
  }

  /** Builder for {@link LiteRtConfig}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setDispatchLibraryPath(String path);

    public abstract Builder setSystemRuntimeHandle(Object handle);

    public abstract LiteRtConfig build();
  }
}
