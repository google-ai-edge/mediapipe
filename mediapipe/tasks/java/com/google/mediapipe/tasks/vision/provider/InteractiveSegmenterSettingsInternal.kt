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
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult
import com.google.mediapipe.tasks.vision.interactivesegmenter.InteractiveSegmenter

/**
 * Internal settings for MediaPipe InteractiveSegmenter task.
 *
 * @property outputConfidenceMasks Whether to output confidence masks.
 * @property outputCategoryMask Whether to output category mask.
 * @property resultListener The listener to receive results.
 * @property errorListener The listener to receive errors.
 */
data class InteractiveSegmenterSettingsInternal
constructor(
  val outputConfidenceMasks: Boolean,
  val outputCategoryMask: Boolean,
  val resultListener: OutputHandler.ResultListener<ImageSegmenterResult, MPImage>?,
  val errorListener: ErrorListener?,
) {
  /**
   * Converts the internal settings to an [InteractiveSegmenter.InteractiveSegmenterOptions].
   *
   * @param baseOptions The base options to use.
   * @return An [InteractiveSegmenter.InteractiveSegmenterOptions] instance.
   */
  fun toOptions(baseOptions: BaseOptions): InteractiveSegmenter.InteractiveSegmenterOptions {
    val optionsBuilder =
      InteractiveSegmenter.InteractiveSegmenterOptions.builder()
        .setBaseOptions(baseOptions)
        .setOutputConfidenceMasks(outputConfidenceMasks)
        .setOutputCategoryMask(outputCategoryMask)
    resultListener?.let { optionsBuilder.setResultListener(it) }
    errorListener?.let { optionsBuilder.setErrorListener(it) }
    return optionsBuilder.build()
  }
}
