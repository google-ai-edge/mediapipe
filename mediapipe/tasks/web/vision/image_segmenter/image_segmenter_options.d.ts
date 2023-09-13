/**
 * Copyright 2022 The MediaPipe Authors.
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
export declare interface ImageSegmenterOptions extends VisionTaskOptions {
  /**
   * The locale to use for display names specified through the TFLite Model
   * Metadata, if any. Defaults to English.
   */
  displayNamesLocale?: string|undefined;

  /** Whether to output confidence masks. Defaults to true. */
  outputConfidenceMasks?: boolean|undefined;

  /** Whether to output the category masks. Defaults to false. */
  outputCategoryMask?: boolean|undefined;
}
