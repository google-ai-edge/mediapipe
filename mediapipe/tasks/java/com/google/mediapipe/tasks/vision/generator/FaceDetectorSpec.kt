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

import com.squareup.kotlinpoet.ClassName
import com.squareup.kotlinpoet.ParameterSpec
import com.squareup.kotlinpoet.ParameterizedTypeName.Companion.parameterizedBy

/** Generates [ParameterSpec] for [FaceDetectorSettingsInternal]. */
object FaceDetectorSpec {
  private val faceDetectorResultCn =
    ClassName("com.google.mediapipe.tasks.vision.facedetector", "FaceDetectorResult")

  /** Gets the [ParameterSpec] list for FaceDetectorSettingsInternal. */
  fun getParams(): List<ParameterSpec> =
    listOf(
      ParameterSpec.builder("minDetectionConfidence", Float::class)
        .defaultValue(Constants.DEFAULT_CONFIDENCE)
        .build(),
      ParameterSpec.builder("minSuppressionThreshold", Float::class)
        .defaultValue(Constants.DEFAULT_CONFIDENCE)
        .build(),
      ParameterSpec.builder("runningMode", Constants.RUNNING_MODE_CLASS)
        .defaultValue(Constants.DEFAULT_RUNNING_MODE)
        .build(),
      ParameterSpec.builder(
          "resultListener",
          Constants.RESULT_LISTENER_BASE_CLASS.parameterizedBy(
              faceDetectorResultCn,
              Constants.MP_IMAGE_CLASS,
            )
            .copy(nullable = true),
        )
        .defaultValue("null")
        .build(),
      ParameterSpec.builder("errorListener", Constants.ERROR_LISTENER_CLASS.copy(nullable = true))
        .defaultValue("null")
        .build(),
    )
}
