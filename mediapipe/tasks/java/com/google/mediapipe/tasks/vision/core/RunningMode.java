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

package com.google.mediapipe.tasks.vision.core;

/**
 * MediaPipe vision task running mode. A MediaPipe vision task can be run with three different
 * modes:
 *
 * <ul>
 *   <li>IMAGE: The mode for running a mediapipe vision task on single image inputs.
 *   <li>VIDEO: The mode for running a mediapipe vision task on the decoded frames of a video.
 *   <li>LIVE_STREAM: The mode for running a mediapipe vision task on a live stream of input data,
 *       such as from camera.
 * </ul>
 */
public enum RunningMode {
  IMAGE,
  VIDEO,
  LIVE_STREAM
}
