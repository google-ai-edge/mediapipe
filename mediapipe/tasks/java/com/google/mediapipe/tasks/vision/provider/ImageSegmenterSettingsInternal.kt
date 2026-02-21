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

import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.ErrorListener
import com.google.mediapipe.tasks.core.OutputHandler
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult

/**
 * Internal settings for MediaPipe ImageSegmenter task.
 *
 * @property outputConfidenceMasks Whether to output confidence masks.
 * @property outputCategoryMask Whether to output a category mask.
 * @property displayNamesLocale The locale to use for display names.
 * @property runningMode The running mode of the task.
 * @property resultListener The listener to receive results.
 * @property errorListener The listener to receive errors.
 */
data class ImageSegmenterSettingsInternal
constructor(
  val outputConfidenceMasks: Boolean,
  val outputCategoryMask: Boolean,
  val displayNamesLocale: String?,
  val runningMode: RunningMode,
  val resultListener: OutputHandler.ResultListener<ImageSegmenterResult, MPImage>?,
  val errorListener: ErrorListener?,
) {
  /**
   * Converts the internal settings to an [ImageSegmenter.ImageSegmenterOptions].
   *
   * @param baseOptions The base options to use.
   * @return An [ImageSegmenter.ImageSegmenterOptions] instance.
   */
  fun toOptions(baseOptions: BaseOptions): ImageSegmenter.ImageSegmenterOptions {
    val optionsBuilder =
      ImageSegmenter.ImageSegmenterOptions.builder()
        .setBaseOptions(baseOptions)
        .setRunningMode(runningMode)
        .setOutputConfidenceMasks(outputConfidenceMasks)
        .setOutputCategoryMask(outputCategoryMask)
        .setDisplayNamesLocale(displayNamesLocale)
    resultListener?.let { optionsBuilder.setResultListener(it) }
    errorListener?.let { optionsBuilder.setErrorListener(it) }
    return optionsBuilder.build()
  }
}
