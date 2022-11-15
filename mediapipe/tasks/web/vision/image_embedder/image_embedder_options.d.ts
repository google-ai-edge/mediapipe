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

import {EmbedderOptions} from '../../../../tasks/web/core/embedder_options';
import {RunningMode} from '../../../../tasks/web/vision/core/running_mode';

/** The options for configuring a MediaPipe image embedder task. */
export declare interface ImageEmbedderOptions extends EmbedderOptions {
  /**
   * The running mode of the task. Default to the image mode.
   * Image embedder has three running modes:
   * 1) The image mode for embedding image on single image inputs.
   * 2) The video mode for embedding image on the decoded frames of a video.
   * 3) The live stream mode for embedding image on the live stream of input
   * data, such as from camera.
   */
  runningMode?: RunningMode;
}
