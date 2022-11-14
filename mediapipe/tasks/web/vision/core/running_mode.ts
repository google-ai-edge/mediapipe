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

/**
 * The running mode of a task.
 * 1) The image mode for processing single image inputs.
 * 2) The video mode for processing decoded frames of a video.
 */
export type RunningMode = 'image'|'video';

/** Configues the `useStreamMode` option . */
export function configureRunningMode(
    options: {runningMode?: RunningMode},
    proto?: BaseOptionsProto): BaseOptionsProto {
  proto = proto ?? new BaseOptionsProto();
  if ('runningMode' in options) {
    const useStreamMode = options.runningMode === 'video';
    proto.setUseStreamMode(useStreamMode);
  }
  return proto;
}
