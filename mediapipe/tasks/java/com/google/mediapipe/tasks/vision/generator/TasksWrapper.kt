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
 * A wrapper class that matches the root structure of the JSON file. It contains a map of all vision
 * task configurations.
 *
 * @property compileSdk The Android compile SDK version.
 * @property minSdk The Android minimum SDK version.
 * @property npuModuleDelivery The delivery mode for NPU modules (e.g., "install-time",
 *   "on-demand").
 * @property deviceTargetingConfiguration The path to the device targeting configuration.
 * @property tasks A map of task names to their [VisionTaskConfig].
 */
data class TasksWrapper(
  @SerializedName("compileSdk") val compileSdk: Int?,
  @SerializedName("minSdk") val minSdk: Int?,
  @SerializedName("npuModuleDelivery") val npuModuleDelivery: String?,
  @SerializedName("deviceTargetingConfiguration") val deviceTargetingConfiguration: String?,
  @SerializedName("tasks") val tasks: Map<String, VisionTaskConfig>?,
)
