/**
 * Copyright 2022 The MediaPipe Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {VisionTaskOptions} from '../../../../tasks/web/vision/core/vision_task_options';

/** Options to configure the MediaPipe Image Segmenter Task */
export interface ImageSegmenterOptions extends VisionTaskOptions {
  /**
   * The locale to use for display names specified through the TFLite Model
   * Metadata, if any. Defaults to English.
   */
  displayNamesLocale?: string|undefined;

  /**
   * The output type of segmentation results.
   *
   * The two supported modes are:
   * - Category Mask:   Gives a single output mask where each pixel represents
   *                    the class which the pixel in the original image was
   *                    predicted to belong to.
   * - Confidence Mask: Gives a list of output masks (one for each class). For
   *                    each mask, the pixel represents the prediction
   *                    confidence, usually in the [0.0, 0.1] range.
   *
   * Defaults to `CATEGORY_MASK`.
   */
  outputType?: 'CATEGORY_MASK'|'CONFIDENCE_MASK'|undefined;
}
