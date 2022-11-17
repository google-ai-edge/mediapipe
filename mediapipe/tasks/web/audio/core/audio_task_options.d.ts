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

import {BaseOptions} from '../../../../tasks/web/core/base_options';

/**
 * MediaPipe audio task running mode. A MediaPipe audio task can be run with
 * two different modes:
 * - audio_clips:  The mode for running a mediapipe audio task on independent
 *                 audio clips.
 * - audio_stream: The mode for running a mediapipe audio task on an audio
 *                 stream, such as from a microphone.
 * </ul>
 */
export type RunningMode = 'audio_clips'|'audio_stream';

/** The options for configuring a MediaPipe Audio Task. */
export declare interface AudioTaskOptions {
  /** Options to configure the loading of the model assets. */
  baseOptions?: BaseOptions;

  /**
   * The running mode of the task. Default to the audio_clips mode.
   * Audio tasks have two running modes:
   * 1) The mode for running a mediapipe audio task on independent
   *    audio clips.
   * 2) The mode for running a mediapipe audio task on an audio
   *    stream, such as from a microphone.
   */
  runningMode?: RunningMode;
}
