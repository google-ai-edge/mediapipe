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
import com.squareup.kotlinpoet.ParameterizedTypeName.Companion.parameterizedBy

/** Constants used across the VisionProvider package. */
object Constants {
  /** The package name for the VisionProvider. */
  const val PACKAGE_NAME = "com.google.mediapipe.tasks.vision.provider"

  // KotlinPoet ClassNames for common types
  val KOTLIN_STRING_CLASS = ClassName("kotlin", "String")
  val RUNNING_MODE_CLASS = ClassName("com.google.mediapipe.tasks.vision.core", "RunningMode")
  val ERROR_LISTENER_CLASS = ClassName("com.google.mediapipe.tasks.core", "ErrorListener")
  val RESULT_LISTENER_BASE_CLASS =
    ClassName("com.google.mediapipe.tasks.core", "OutputHandler", "ResultListener")
  val MP_IMAGE_CLASS = ClassName("com.google.mediapipe.framework.image", "MPImage")
  val STRING_LIST_TYPE =
    KOTLIN_STRING_CLASS.let { ClassName("kotlin.collections", "List").parameterizedBy(it) }
  val CLASSIFIER_OPTIONS_CLASS =
    ClassName("com.google.mediapipe.tasks.components.processors", "ClassifierOptions")
  val CONSTANTS_CLASS = ClassName(PACKAGE_NAME, "Constants")
  val VISION_PROVIDER_BASE_CLASS = ClassName(PACKAGE_NAME, "VisionProviderBase")
  val NON_NULL_CLASS = ClassName("androidx.annotation", "NonNull")

  val UNLIMITED_RESULTS = CodeBlock.of("%L", -1)
  val DEFAULT_NUM_FACES = CodeBlock.of("%L", 1)
  val DEFAULT_NUM_HANDS = CodeBlock.of("%L", 1)
  val DEFAULT_OUTPUT_BLENDSHAPES = CodeBlock.of("%L", false)
  val DEFAULT_OUTPUT_FACIAL_TRANSFORMATION_MATRIXES = CodeBlock.of("%L", false)
  val DEFAULT_DISPLAY_NAMES_LOCALE = CodeBlock.of("%S", "en")
  val DEFAULT_L2_NORMALIZE = CodeBlock.of("%L", false)
  val DEFAULT_CONFIDENCE = CodeBlock.of("%Lf", 0.5f)
  val DEFAULT_QUANTIZE = CodeBlock.of("%L", false)
  val DEFAULT_OUTPUT_CONFIDENCE_MASKS = CodeBlock.of("%L", false)
  val DEFAULT_OUTPUT_CATEGORY_MASK = CodeBlock.of("%L", true)
  val DEFAULT_OUTPUT_SEGMENTATION_MASKS = CodeBlock.of("%L", false)
  val DEFAULT_RUNNING_MODE = CodeBlock.of("%T.IMAGE", RUNNING_MODE_CLASS)
}
