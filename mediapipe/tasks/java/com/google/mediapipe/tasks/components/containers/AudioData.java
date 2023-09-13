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

package com.google.mediapipe.tasks.components.containers;

import static java.lang.System.arraycopy;

import android.media.AudioFormat;
import android.media.AudioRecord;
import com.google.auto.value.AutoValue;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Defines a ring buffer and some utility functions to prepare the input audio samples.
 *
 * <p>It maintains a <a href="https://en.wikipedia.org/wiki/Circular_buffer">Ring Buffer</a> to hold
 * input audio data. Clients could feed input audio data via `load` methods and access the
 * aggregated audio samples via `getTensorBuffer` method.
 *
 * <p>Note that this class can only handle input audio in Float (in {@link
 * android.media.AudioFormat#ENCODING_PCM_16BIT}) or Short (in {@link
 * android.media.AudioFormat#ENCODING_PCM_FLOAT}). Internally it converts and stores all the audio
 * samples in PCM Float encoding.
 *
 * <p>Typical usage in Kotlin
 *
 * <pre>
 *   val audioData = AudioData.create(format, modelInputLength)
 *   audioData.load(newData)
 * </pre>
 *
 * <p>Another sample usage with {@link android.media.AudioRecord}
 *
 * <pre>
 *   val audioData = AudioData.create(format, modelInputLength)
 *   Timer().scheduleAtFixedRate(delay, period) {
 *     audioData.load(audioRecord)
 *   }
 * </pre>
 */
public class AudioData {

  private static final String TAG = AudioData.class.getSimpleName();
  private final FloatRingBuffer buffer;
  private final AudioDataFormat format;

  /**
   * Creates a {@link android.media.AudioRecord} instance with a ring buffer whose size is {@code
   * sampleCounts} * {@code format.getNumOfChannels()}.
   *
   * @param format the expected {@link AudioDataFormat} of audio data loaded into this class.
   * @param sampleCounts the number of samples.
   */
  public static AudioData create(AudioDataFormat format, int sampleCounts) {
    return new AudioData(format, sampleCounts);
  }

  /**
   * Creates a {@link AudioData} instance with a ring buffer whose size is {@code sampleCounts} *
   * {@code format.getChannelCount()}.
   *
   * @param format the {@link android.media.AudioFormat} required by the TFLite model. It defines
   *     the number of channels and sample rate.
   * @param sampleCounts the number of samples to be fed into the model
   */
  public static AudioData create(AudioFormat format, int sampleCounts) {
    return new AudioData(AudioDataFormat.create(format), sampleCounts);
  }

  /**
   * Wraps a few constants describing the format of the incoming audio samples, namely number of
   * channels and the sample rate. By default, num of channels is set to 1.
   */
  @AutoValue
  public abstract static class AudioDataFormat {
    private static final int DEFAULT_NUM_OF_CHANNELS = 1;

    /** Creates a {@link AudioFormat} instance from Android AudioFormat class. */
    public static AudioDataFormat create(AudioFormat format) {
      return AudioDataFormat.builder()
          .setNumOfChannels(format.getChannelCount())
          .setSampleRate(format.getSampleRate())
          .build();
    }

    public abstract int getNumOfChannels();

    public abstract float getSampleRate();

    public static Builder builder() {
      return new AutoValue_AudioData_AudioDataFormat.Builder()
          .setNumOfChannels(DEFAULT_NUM_OF_CHANNELS);
    }

    /** Builder for {@link AudioDataFormat} */
    @AutoValue.Builder
    public abstract static class Builder {

      /* By default, it's set to have 1 channel. */
      public abstract Builder setNumOfChannels(int value);

      public abstract Builder setSampleRate(float value);

      abstract AudioDataFormat autoBuild();

      public AudioDataFormat build() {
        AudioDataFormat format = autoBuild();
        if (format.getNumOfChannels() <= 0) {
          throw new IllegalArgumentException("Number of channels should be greater than 0");
        }
        if (format.getSampleRate() <= 0) {
          throw new IllegalArgumentException("Sample rate should be greater than 0");
        }
        return format;
      }
    }
  }

  /**
   * Stores the input audio samples {@code src} in the ring buffer.
   *
   * @param src input audio samples in {@link android.media.AudioFormat#ENCODING_PCM_FLOAT}. For
   *     multi-channel input, the array is interleaved.
   */
  public void load(float[] src) {
    load(src, 0, src.length);
  }

  /**
   * Stores the input audio samples {@code src} in the ring buffer.
   *
   * @param src input audio samples in {@link android.media.AudioFormat#ENCODING_PCM_FLOAT}. For
   *     multi-channel input, the array is interleaved.
   * @param offsetInFloat starting position in the {@code src} array
   * @param sizeInFloat the number of float values to be copied
   * @throws IllegalArgumentException for incompatible audio format or incorrect input size
   */
  public void load(float[] src, int offsetInFloat, int sizeInFloat) {
    if (sizeInFloat % format.getNumOfChannels() != 0) {
      throw new IllegalArgumentException(
          String.format(
              "Size (%d) needs to be a multiplier of the number of channels (%d)",
              sizeInFloat, format.getNumOfChannels()));
    }
    buffer.load(src, offsetInFloat, sizeInFloat);
  }

  /**
   * Converts the input audio samples {@code src} to ENCODING_PCM_FLOAT, then stores it in the ring
   * buffer.
   *
   * @param src input audio samples in {@link android.media.AudioFormat#ENCODING_PCM_16BIT}. For
   *     multi-channel input, the array is interleaved.
   */
  public void load(short[] src) {
    load(src, 0, src.length);
  }

  /**
   * Converts the input audio samples {@code src} to ENCODING_PCM_FLOAT, then stores it in the ring
   * buffer.
   *
   * @param src input audio samples in {@link android.media.AudioFormat#ENCODING_PCM_16BIT}. For
   *     multi-channel input, the array is interleaved.
   * @param offsetInShort starting position in the src array
   * @param sizeInShort the number of short values to be copied
   * @throws IllegalArgumentException if the source array can't be copied
   */
  public void load(short[] src, int offsetInShort, int sizeInShort) {
    if (offsetInShort + sizeInShort > src.length) {
      throw new IllegalArgumentException(
          String.format(
              "Index out of range. offset (%d) + size (%d) should <= newData.length (%d)",
              offsetInShort, sizeInShort, src.length));
    }
    float[] floatData = new float[sizeInShort];
    for (int i = 0; i < sizeInShort; i++) {
      // Convert the data to PCM Float encoding i.e. values between -1 and 1
      floatData[i] = src[i + offsetInShort] * 1.f / Short.MAX_VALUE;
    }
    load(floatData);
  }

  /**
   * Loads latest data from the {@link android.media.AudioRecord} in a non-blocking way. Only
   * supporting ENCODING_PCM_16BIT and ENCODING_PCM_FLOAT.
   *
   * @param record an instance of {@link android.media.AudioRecord}
   * @return number of captured audio values whose size is {@code channelCount * sampleCount}. If
   *     there was no new data in the AudioRecord or an error occurred, this method will return 0.
   * @throws IllegalArgumentException for unsupported audio encoding format
   * @throws IllegalStateException if reading from AudioRecord failed
   */
  public int load(AudioRecord record) {
    if (!this.format.equals(AudioDataFormat.create(record.getFormat()))) {
      throw new IllegalArgumentException("Incompatible audio format.");
    }
    int loadedValues = 0;
    if (record.getAudioFormat() == AudioFormat.ENCODING_PCM_FLOAT) {
      float[] newData = new float[record.getChannelCount() * record.getBufferSizeInFrames()];
      loadedValues = record.read(newData, 0, newData.length, AudioRecord.READ_NON_BLOCKING);
      if (loadedValues > 0) {
        load(newData, 0, loadedValues);
        return loadedValues;
      }
    } else if (record.getAudioFormat() == AudioFormat.ENCODING_PCM_16BIT) {
      short[] newData = new short[record.getChannelCount() * record.getBufferSizeInFrames()];
      loadedValues = record.read(newData, 0, newData.length, AudioRecord.READ_NON_BLOCKING);
      if (loadedValues > 0) {
        load(newData, 0, loadedValues);
        return loadedValues;
      }
    } else {
      throw new IllegalArgumentException(
          "Unsupported encoding. Requires ENCODING_PCM_16BIT or ENCODING_PCM_FLOAT.");
    }

    switch (loadedValues) {
      case AudioRecord.ERROR_INVALID_OPERATION:
        throw new IllegalStateException("AudioRecord.ERROR_INVALID_OPERATION");

      case AudioRecord.ERROR_BAD_VALUE:
        throw new IllegalStateException("AudioRecord.ERROR_BAD_VALUE");

      case AudioRecord.ERROR_DEAD_OBJECT:
        throw new IllegalStateException("AudioRecord.ERROR_DEAD_OBJECT");

      case AudioRecord.ERROR:
        throw new IllegalStateException("AudioRecord.ERROR");

      default:
        return 0;
    }
  }

  /**
   * Returns a float array holding all the available audio samples in {@link
   * android.media.AudioFormat#ENCODING_PCM_FLOAT} i.e. values are in the range of [-1, 1].
   */
  public float[] getBuffer() {
    float[] bufferData = new float[buffer.getCapacity()];
    ByteBuffer byteBuffer = buffer.getBuffer();
    byteBuffer.asFloatBuffer().get(bufferData);
    return bufferData;
  }

  /* Returns the {@link AudioDataFormat} associated with the tensor. */
  public AudioDataFormat getFormat() {
    return format;
  }

  /* Returns the audio buffer length. */
  public int getBufferLength() {
    return buffer.getCapacity() / format.getNumOfChannels();
  }

  private AudioData(AudioDataFormat format, int sampleCounts) {
    this.format = format;
    this.buffer = new FloatRingBuffer(sampleCounts * format.getNumOfChannels());
  }

  /** Actual implementation of the ring buffer. */
  private static class FloatRingBuffer {

    private final float[] buffer;
    private int nextIndex = 0;

    public FloatRingBuffer(int flatSize) {
      buffer = new float[flatSize];
    }

    /**
     * Loads a slice of the float array to the ring buffer. If the float array is longer than ring
     * buffer's capacity, samples with lower indices in the array will be ignored.
     */
    public void load(float[] newData, int offset, int size) {
      if (offset + size > newData.length) {
        throw new IllegalArgumentException(
            String.format(
                "Index out of range. offset (%d) + size (%d) should <= newData.length (%d)",
                offset, size, newData.length));
      }
      // If buffer can't hold all the data, only keep the most recent data of size buffer.length
      if (size > buffer.length) {
        offset += (size - buffer.length);
        size = buffer.length;
      }
      if (nextIndex + size < buffer.length) {
        // No need to wrap nextIndex, just copy newData[offset:offset + size]
        // to buffer[nextIndex:nextIndex+size]
        arraycopy(newData, offset, buffer, nextIndex, size);
      } else {
        // Need to wrap nextIndex, perform copy in two chunks.
        int firstChunkSize = buffer.length - nextIndex;
        // First copy newData[offset:offset+firstChunkSize] to buffer[nextIndex:buffer.length]
        arraycopy(newData, offset, buffer, nextIndex, firstChunkSize);
        // Then copy newData[offset+firstChunkSize:offset+size] to buffer[0:size-firstChunkSize]
        arraycopy(newData, offset + firstChunkSize, buffer, 0, size - firstChunkSize);
      }

      nextIndex = (nextIndex + size) % buffer.length;
    }

    public ByteBuffer getBuffer() {
      // Create non-direct buffers. On Pixel 4, creating direct buffer costs around 0.1 ms, which
      // can be 5x ~ 10x longer compared to non-direct buffer backed by arrays (around 0.01ms), so
      // generally we don't create direct buffer for every invocation.
      ByteBuffer byteBuffer = ByteBuffer.allocate(Float.SIZE / 8 * buffer.length);
      byteBuffer.order(ByteOrder.nativeOrder());
      FloatBuffer result = byteBuffer.asFloatBuffer();
      result.put(buffer, nextIndex, buffer.length - nextIndex);
      result.put(buffer, 0, nextIndex);
      byteBuffer.rewind();
      return byteBuffer;
    }

    public int getCapacity() {
      return buffer.length;
    }
  }
}
