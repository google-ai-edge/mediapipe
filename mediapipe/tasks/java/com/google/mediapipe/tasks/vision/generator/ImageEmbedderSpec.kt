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
import com.squareup.kotlinpoet.CodeBlock
import com.squareup.kotlinpoet.ParameterSpec
import com.squareup.kotlinpoet.ParameterizedTypeName.Companion.parameterizedBy

/** Generates [ParameterSpec] for [ImageEmbedderSettingsInternal]. */
object ImageEmbedderSpec {
  private val imageEmbedderResultCn =
    ClassName("com.google.mediapipe.tasks.vision.imageembedder", "ImageEmbedderResult")

  /** Gets the [ParameterSpec] list for [ImageEmbedderSettingsInternal]. */
  fun getParams(): List<ParameterSpec> =
    listOf(
      ParameterSpec.builder("l2Normalize", Boolean::class)
        .defaultValue(CodeBlock.of("%L", false))
        .build(),
      ParameterSpec.builder("quantize", Boolean::class)
        .defaultValue(CodeBlock.of("%L", false))
        .build(),
      ParameterSpec.builder("runningMode", Constants.RUNNING_MODE_CLASS)
        .defaultValue(Constants.DEFAULT_RUNNING_MODE)
        .build(),
      ParameterSpec.builder(
          "resultListener",
          Constants.RESULT_LISTENER_BASE_CLASS.parameterizedBy(
              imageEmbedderResultCn,
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
