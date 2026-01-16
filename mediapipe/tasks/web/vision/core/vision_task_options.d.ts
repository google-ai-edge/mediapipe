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

import {TaskRunnerOptions} from '../../../../tasks/web/core/task_runner_options';

/**
 * The two running modes of a vision task.
 * 1) The image mode for processing single image inputs.
 * 2) The video mode for processing decoded frames of a video.
 */
export type RunningMode = 'IMAGE'|'VIDEO';

/** The options for configuring a MediaPipe vision task. */
export declare interface VisionTaskOptions extends TaskRunnerOptions {
  /**
   * The canvas element to bind textures to. This has to be set for GPU
   * processing. The task will initialize a WebGL context and throw an error if
   * this fails (e.g. if you have already initialized a different type of
   * context).
   */
  canvas?: HTMLCanvasElement|OffscreenCanvas;

  /**
   * The running mode of the task. Default to the image mode.
   * Vision tasks have two running modes:
   * 1) The image mode for processing single image inputs.
   * 2) The video mode for processing decoded frames of a video.
   */
  runningMode?: RunningMode;
}
