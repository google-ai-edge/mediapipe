// Copyright 2019 The MediaPipe Authors.
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

package com.google.mediapipe.components;

import android.media.AudioFormat;
import java.nio.ByteBuffer;

/** Lightweight abstraction for an object that can receive audio data. */
public interface AudioDataConsumer {
  /**
   * Called when a new audio data buffer is available. Note, for consistency, the ByteBuffer used in
   * AudioDataConsumer has to use AudioFormat.ENCODING_PCM_16BIT, 2 bytes per sample, FILLED with
   * ByteOrder.LITTLE_ENDIAN, which is ByteOrder.nativeOrder() on Android
   * (https://developer.android.com/ndk/guides/abis.html).
   */
  public abstract void onNewAudioData(
      ByteBuffer audioData, long timestampMicros, AudioFormat audioFormat);
}
