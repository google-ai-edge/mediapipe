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

import {TaskRunner} from '../../../../tasks/web/core/task_runner';
import {TaskRunnerOptions} from '../../../../tasks/web/core/task_runner_options';
import {WasmFileset} from '../../../../tasks/web/core/wasm_fileset';
import {WasmMediaPipeConstructor} from '../../../../web/graph_runner/graph_runner';


/** Base class for all MediaPipe Audio Tasks. */
export abstract class AudioTaskRunner<T> extends TaskRunner {
  private defaultSampleRate = 48000;

  protected static async createAudioInstance<T, I extends AudioTaskRunner<T>>(
      type: WasmMediaPipeConstructor<I>, fileset: WasmFileset,
      options: TaskRunnerOptions): Promise<I> {
    return TaskRunner.createInstance(
        type, /* canvas= */ null, fileset, options);
  }

  /**
   * Sets the sample rate for API calls that omit an explicit sample rate.
   * `48000` is used as a default if this method is not called.
   *
   * @export
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
        audioData, sampleRate ?? this.defaultSampleRate,
        this.getSynctheticTimestamp());
  }
}


