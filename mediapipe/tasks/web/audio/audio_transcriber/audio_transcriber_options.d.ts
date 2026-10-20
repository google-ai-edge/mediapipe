/**
 * Copyright 2026 The MediaPipe Authors.
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
 * Model architecture family for AudioTranscriber.
 */
export type AudioTranscriberModelArchitecture = 'AUTO' | 'OMNI' | 'TINY_GEMMA';

/**
 * Options to configure the MediaPipe AudioTranscriber Task on the Web.
 */
export declare interface AudioTranscriberOptions extends TaskRunnerOptions {
  /**
   * Input audio sample rate in Hz. Defaults to 16000 Hz when unset or 0.
   */
  sampleRateHz?: number;

  /**
   * Whether to emit interim (non-final) transcription segments during
   * streaming. Defaults to false.
   */
  enableInterimResults?: boolean;

  /**
   * Model architecture backend family ('AUTO', 'OMNI', or 'TINY_GEMMA', or
   * numeric code 0=AUTO, 1=OMNI, 2=TINY_GEMMA). Defaults to 'AUTO'.
   */
  modelArchitecture?: AudioTranscriberModelArchitecture | number;

  /**
   * Optional system or user prompt override for generative ASR models.
   */
  prompt?: string;

  /**
   * Optional directory path in the Wasm virtual filesystem for caching
   * compiled model artifacts.
   */
  cacheDir?: string;

  /**
   * Optional byte length of `baseOptions.modelAssetBuffer` when provided as a
   * `ReadableStreamDefaultReader`, enabling single-pass zero-copy streaming
   * allocation directly in the Wasm heap.
   */
  modelAssetSize?: number;
}
