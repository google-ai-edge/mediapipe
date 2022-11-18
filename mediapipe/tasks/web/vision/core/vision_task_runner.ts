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

import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {convertBaseOptionsToProto} from '../../../../tasks/web/components/processors/base_options';
import {TaskRunner} from '../../../../tasks/web/core/task_runner';
import {ImageSource} from '../../../../web/graph_runner/graph_runner';

import {VisionTaskOptions} from './vision_task_options';

/** Base class for all MediaPipe Vision Tasks. */
export abstract class VisionTaskRunner<T> extends TaskRunner {
  protected abstract baseOptions?: BaseOptionsProto|undefined;

  /** Configures the shared options of a vision task. */
  async setOptions(options: VisionTaskOptions): Promise<void> {
    this.baseOptions = this.baseOptions ?? new BaseOptionsProto();
    if (options.baseOptions) {
      this.baseOptions = await convertBaseOptionsToProto(
          options.baseOptions, this.baseOptions);
    }
    if ('runningMode' in options) {
      const useStreamMode =
          !!options.runningMode && options.runningMode !== 'image';
      this.baseOptions.setUseStreamMode(useStreamMode);
    }
  }

  /** Sends an image packet to the graph and awaits results. */
  protected abstract process(input: ImageSource, timestamp: number): T;

  /** Sends a single image to the graph and awaits results. */
  protected processImageData(image: ImageSource): T {
    if (!!this.baseOptions?.getUseStreamMode()) {
      throw new Error(
          'Task is not initialized with image mode. ' +
          '\'runningMode\' must be set to \'image\'.');
    }
    return this.process(image, performance.now());
  }

  /** Sends a single video frame to the graph and awaits results. */
  protected processVideoData(imageFrame: ImageSource, timestamp: number): T {
    if (!this.baseOptions?.getUseStreamMode()) {
      throw new Error(
          'Task is not initialized with video mode. ' +
          '\'runningMode\' must be set to \'video\'.');
    }
    return this.process(imageFrame, timestamp);
  }
}


