/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.mediapipe.tasks.audio.core;

import com.google.mediapipe.tasks.components.containers.AudioData;
import com.google.mediapipe.tasks.core.Part;

/** Represents an audio part of a multi-modal content block. */
public final class AudioPart extends Part {
  private final AudioData audioData;

  /**
   * Creates a new {@link AudioPart} with the specified audio data.
   *
   * @param audioData the audio data content.
   */
  public AudioPart(AudioData audioData) {
    this.audioData = audioData;
  }

  /**
   * Returns the audio data content of this part.
   *
   * @return the audio data.
   */
  public AudioData audioData() {
    return audioData;
  }
}
