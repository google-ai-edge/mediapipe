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

package com.google.mediapipe.tasks.core;

import static java.nio.charset.StandardCharsets.US_ASCII;

import android.net.Uri;
import androidx.annotation.Nullable;
import com.google.mediapipe.tasks.components.containers.AudioData;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Represents an audio part of a multi-modal content block. */
public final class AudioPart extends Part {
  private static final int WAV_HEADER_SIZE = 44;

  @Nullable private final Uri filePath;
  @Nullable private final AudioData audioData;

  /**
   * Creates a new {@link AudioPart} with the specified audio file path.
   *
   * @param filePath the audio file path.
   */
  public AudioPart(@Nullable Uri filePath) {
    this(filePath, null);
  }

  /**
   * Creates a new {@link AudioPart} with the specified in-memory {@link AudioData}.
   *
   * @param audioData the decoded audio data.
   */
  public AudioPart(AudioData audioData) {
    this(null, audioData);
  }

  /**
   * Creates a new {@link AudioPart} with the specified file path and decoded audio data.
   *
   * @param filePath the optional audio file path.
   * @param audioData the optional decoded audio data.
   */
  public AudioPart(@Nullable Uri filePath, @Nullable AudioData audioData) {
    this.filePath = filePath;
    this.audioData = audioData;
  }

  /**
   * Returns {@code true} if the given byte array starts with a valid 12-byte RIFF/WAVE header.
   *
   * @param bytes the byte array to inspect.
   * @return whether the byte array has a RIFF/WAVE header.
   */
  public static boolean isWavHeader(@Nullable byte[] bytes) {
    return bytes != null
        && bytes.length >= 12
        && new String(bytes, 0, 4, US_ASCII).equals("RIFF")
        && new String(bytes, 8, 4, US_ASCII).equals("WAVE");
  }

  /**
   * Decodes 16-bit PCM WAV bytes into {@link AudioData} and returns a new {@link AudioPart}.
   *
   * @param wavBytes the encoded WAV audio bytes.
   * @return a new {@link AudioPart} containing the decoded {@link AudioData}.
   */
  @SuppressWarnings("PreferPreconditions")
  public static AudioPart fromWavBytes(byte[] wavBytes) {
    if (wavBytes == null || wavBytes.length < WAV_HEADER_SIZE) {
      throw new IllegalArgumentException("WAV bytes must contain at least a 44-byte header.");
    }
    if (!isWavHeader(wavBytes)) {
      throw new IllegalArgumentException("Invalid RIFF/WAVE header in wavBytes.");
    }
    ByteBuffer header =
        ByteBuffer.wrap(wavBytes, 0, WAV_HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN);
    int channels = Short.toUnsignedInt(header.getShort(22));
    int sampleRate = header.getInt(24);
    int bitsPerSample = Short.toUnsignedInt(header.getShort(34));
    if (channels <= 0 || sampleRate <= 0 || bitsPerSample != 16) {
      throw new IllegalArgumentException(
          "Unsupported WAV format: expected 16-bit PCM with positive sample rate and channels.");
    }
    int pcmByteLength = wavBytes.length - WAV_HEADER_SIZE;
    int totalSamples = pcmByteLength / 2;
    int frameCount = Math.max(1, totalSamples / channels);
    float[] floatSamples = new float[totalSamples];
    ByteBuffer pcmBuffer =
        ByteBuffer.wrap(wavBytes, WAV_HEADER_SIZE, totalSamples * 2).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < totalSamples; i++) {
      floatSamples[i] = pcmBuffer.getShort() / 32768.0f;
    }
    AudioData.AudioDataFormat format =
        AudioData.AudioDataFormat.builder()
            .setNumOfChannels(channels)
            .setSampleRate(sampleRate)
            .build();
    AudioData audioData = AudioData.create(format, frameCount);
    if (totalSamples > 0) {
      audioData.load(floatSamples);
    }
    return new AudioPart(null, audioData);
  }

  /**
   * Returns the file path of this audio, or {@code null} if stored in-memory.
   *
   * @return the file path.
   */
  @Nullable
  public Uri getFilePath() {
    return filePath;
  }

  /**
   * Returns the file path of this audio, or {@code null} if stored in-memory.
   *
   * @return the file path.
   */
  @Nullable
  public Uri filePath() {
    return filePath;
  }

  /**
   * Returns the decoded in-memory {@link AudioData} of this audio, or {@code null} if not present.
   *
   * @return the decoded audio data.
   */
  @Nullable
  public AudioData audioData() {
    return audioData;
  }
}
