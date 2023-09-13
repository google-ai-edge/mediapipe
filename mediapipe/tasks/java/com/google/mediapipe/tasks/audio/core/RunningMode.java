// Copyright 2022 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.tasks.audio.core;

/**
 * MediaPipe audio task running mode. A MediaPipe audio task can be run with two different modes:
 *
 * <ul>
 *   <li>AUDIO_CLIPS: The mode for running a mediapipe audio task on independent audio clips.
 *   <li>AUDIO_STREAM: The mode for running a mediapipe audio task on an audio stream, such as from
 *       a microphone.
 * </ul>
 */
public enum RunningMode {
  AUDIO_CLIPS,
  AUDIO_STREAM,
}
