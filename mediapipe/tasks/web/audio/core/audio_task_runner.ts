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

import {AudioTaskOptions} from './audio_task_options';

/** Base class for all MediaPipe Audio Tasks. */
export abstract class AudioTaskRunner<T> extends TaskRunner {
  protected abstract baseOptions?: BaseOptionsProto|undefined;
  private defaultSampleRate = 48000;

  /** Configures the shared options of an audio task. */
  async setOptions(options: AudioTaskOptions): Promise<void> {
    this.baseOptions = this.baseOptions ?? new BaseOptionsProto();
    if (options.baseOptions) {
      this.baseOptions = await convertBaseOptionsToProto(
          options.baseOptions, this.baseOptions);
    }
  }

  /**
   * Sets the sample rate for API calls that omit an explicit sample rate.
   * `48000` is used as a default if this method is not called.
   *
   * @param sampleRate A sample rate (e.g. `44100`).
   */
  setDefaultSampleRate(sampleRate: number) {
    this.defaultSampleRate = sampleRate;
  }

  /** Sends an audio packet to the graph and awaits results. */
  protected abstract process(
      audioData: Float32Array, sampleRate: number, timestampMs: number): T;

  /** Sends a single audio clip to the graph and awaits results. */
  protected processAudioClip(audioData: Float32Array, sampleRate?: number): T {
    return this.process(
        audioData, sampleRate ?? this.defaultSampleRate, performance.now());
  }
}


