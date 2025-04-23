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

import {ClassifierOptions} from '../../../../tasks/web/core/classifier_options';
import {VisionTaskOptions} from '../../../../tasks/web/vision/core/vision_task_options';

/** Options to configure the MediaPipe Object Detector Task */
export declare interface ObjectDetectorOptions extends VisionTaskOptions,
                                                       ClassifierOptions {
  /**
   * Whether to use multiclass non-maximum suppression. That is, each category
   * processes non-maximum-suppression separately.
   */
  multiclassNms?: boolean;
}
