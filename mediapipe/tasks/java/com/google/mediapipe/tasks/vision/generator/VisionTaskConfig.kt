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

package com.google.mediapipe.tasks.vision.generator

import com.google.gson.annotations.SerializedName

/**
 * Represents the configuration for a single MediaPipe vision task.
 *
 * @property defaultModel The name of the default model to use for this task.
 * @property models A list of available model names for this task.
 * @property delivery The delivery type for the AI Pack associated with this task (e.g.,
 *   "install-time").
 */
data class VisionTaskConfig(
  @SerializedName("defaultModel") val defaultModel: String?,
  @SerializedName("models") val models: List<String>?,
  @SerializedName("delivery") val delivery: String?,
)
