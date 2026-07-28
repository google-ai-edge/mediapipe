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

package com.google.mediapipe.tasks.core;

/**
 * Build-time configuration for upstream MediaPipe Tasks JNI library names.
 *
 * <p>TODO: gkarpiak - migrate all tasks to use a dedicated method to load libs.
 */
public final class JniConfig {
  public static final JniConfigContract INSTANCE = new DefaultJniConfig();

  private static final class DefaultJniConfig implements JniConfigContract {
    @Override
    public String getVisionJniLib() {
      return "mediapipe_tasks_jni";
    }
  }

  private JniConfig() {}
}
