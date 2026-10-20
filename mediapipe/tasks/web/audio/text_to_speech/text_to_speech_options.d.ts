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
 * Per-utterance synthesis overrides passed to `TextToSpeech.synthesize()` and
 * `TextToSpeech.synthesizeStream()`.
 */
export declare interface SynthesisOptions {
  /** Optional voice identifier override (e.g. "af_heart", "ef_dora"). */
  voiceId?: string;
  /** Optional BCP-47 locale override (e.g. "en-US", "es-ES", "ja-JP"). */
  locale?: string;
}

/**
 * Options to configure the MediaPipe TextToSpeech Task on the Web.
 */
export declare interface TextToSpeechOptions extends TaskRunnerOptions {
  /**
   * Optional default voice ID (e.g. "af_heart"). When unset, the engine's
   * default voice is used.
   */
  defaultVoiceId?: string;

  /**
   * Optional default BCP-47 locale (e.g. "en-US"). When unset, inferred from
   * the selected voice metadata.
   */
  defaultLocale?: string;

  /**
   * Speaking rate multiplier in `(0.0, 4.0]`. Defaults to `1.0` (normal speed).
   */
  speakingRate?: number;

  /**
   * Target sequence length bucket for static-shape Kokoro models (`0` = auto).
   */
  targetBucket?: number;

  /**
   * Optional directory path in the Wasm virtual filesystem for caching unpacked
   * `.litertlm` assets and compiled weights.
   */
  cacheDirectory?: string;

  /**
   * Optional byte length of `baseOptions.modelAssetBuffer` when provided as a
   * `ReadableStreamDefaultReader`, enabling single-pass zero-copy streaming
   * allocation directly in the Wasm heap.
   */
  modelAssetSize?: number;
}
