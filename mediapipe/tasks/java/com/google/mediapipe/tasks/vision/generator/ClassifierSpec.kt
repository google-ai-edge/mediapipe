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

import com.squareup.kotlinpoet.CodeBlock
import com.squareup.kotlinpoet.ParameterSpec

/** Generates [ParameterSpec] for [ClassifierSettingsInternal]. */
object ClassifierSpec {

  /** Gets the [ParameterSpec] list for [ClassifierSettingsInternal]. */
  fun getParams(): List<ParameterSpec> =
    listOf(
      ParameterSpec.builder("displayNamesLocale", String::class)
        .defaultValue(Constants.DEFAULT_DISPLAY_NAMES_LOCALE)
        .build(),
      ParameterSpec.builder("maxResults", Int::class)
        .defaultValue(Constants.UNLIMITED_RESULTS)
        .build(),
      ParameterSpec.builder("scoreThreshold", Float::class)
        .defaultValue(CodeBlock.of("%Lf", 0.0f))
        .build(),
      ParameterSpec.builder("categoryAllowlist", Constants.STRING_LIST_TYPE)
        .defaultValue(CodeBlock.of("emptyList()"))
        .build(),
      ParameterSpec.builder("categoryDenylist", Constants.STRING_LIST_TYPE)
        .defaultValue(CodeBlock.of("emptyList()"))
        .build(),
    )
}
