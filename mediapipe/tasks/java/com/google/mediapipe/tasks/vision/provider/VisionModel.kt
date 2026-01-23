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

package com.google.mediapipe.tasks.vision.provider

import java.util.regex.Pattern

/**
 * The public definition of a vision model.
 *
 * @property task The [VisionTask.Type] associated with this model.
 * @property enumName The uppercase name of the model as used in enums.
 * @property modelName The base name of the model.
 * @property version The version of the model.
 * @property quantization The [Quantization] type of the model.
 */
interface VisionModel {

  val task: VisionTask.Type
  val enumName: String
  val modelName: String
  val version: String
  val quantization: Quantization

  /**
   * Creates the model file name including the extension.
   *
   * @return The model file name as a string.
   */
  fun createModelFileName(): String {
    val extension = task.modelExtension
    return "${modelName}.$extension"
  }

  /**
   * Gets a list of supported variants for this model.
   *
   * @return A list of strings representing the model variants.
   */
  fun getVariants(): List<String> {
    return listOf(
      "other",
      "mediatek_mt6878_android15",
      "mediatek_mt6897_android15",
      "mediatek_mt6983_android15",
      "mediatek_mt6985_android15",
      "mediatek_mt6989_android15",
      "mediatek_mt6991_android15",
      "qualcomm_sm8350",
      "qualcomm_sm8450",
      "qualcomm_sm8650",
      "qualcomm_sm8750",
    )
  }

  /**
   * Creates a list of URLs for downloading the model variants from GCS.
   *
   * @return A list of strings, each representing a URL to a model variant.
   */
  fun createModelUrls(): List<String> {
    val modelVariants = getVariants()
    val versionNumber = version.substring(1)
    val quantizationStr = quantization.gcsDirectoryName
    val taskPath = task.name.lowercase()

    val baseUrl =
      "https://storage.googleapis.com/mediapipe-models/$taskPath/$modelName/$quantizationStr/$versionNumber/"
    return modelVariants.map { variant ->
      val modelNameWithVariant = if (variant != "other") "${modelName}_$variant" else modelName
      val fileName = "$modelNameWithVariant.${task.modelExtension}"
      "$baseUrl$fileName"
    }
  }

  companion object {
    /**
     * Creates a [VisionModel] instance from a canonical name. The canonical name is expected to be
     * in the format "modelname_vX_quantization", e.g., "facedetector_v1_fp16".
     *
     * @param task The [VisionTask.Type] associated with the model.
     * @param name The canonical name of the model.
     * @return A [VisionModel] instance.
     * @throws IllegalArgumentException if the name does not match the expected format.
     */
    fun fromCanonicalName(task: VisionTask.Type, name: String): VisionModel {
      val pattern = Pattern.compile("^(.*)_(v\\d+)_(fp16|fp32|int8)$")
      val matcher = pattern.matcher(name)
      val modelName: String
      val version: String
      val quantization: Quantization
      if (matcher.matches()) {
        modelName = matcher.group(1)!!
        version = matcher.group(2)!!
        quantization = Quantization.fromCanonicalName(matcher.group(3)!!)
      } else {
        throw IllegalArgumentException("Invalid model name: $name")
      }
      return object : VisionModel {
        override val task: VisionTask.Type = task
        override val enumName: String = name.uppercase().replace('.', '_')
        override val modelName: String = modelName
        override val version: String = version
        override val quantization: Quantization = quantization
      }
    }
  }
}
