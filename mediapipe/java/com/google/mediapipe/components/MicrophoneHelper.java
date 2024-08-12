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

import static java.lang.Math.max;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTimestamp;
import android.media.MediaRecorder.AudioSource;
import android.os.Build.VERSION;
import android.os.Build.VERSION_CODES;
import android.util.Log;
import com.google.common.base.Preconditions;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.CopyOnWriteArraySet;

/** Provides access to audio data from a microphone. */
public class MicrophoneHelper implements AudioDataProducer {
  private static final String TAG = "MicrophoneHelper";

  private static final int AUDIO_ENCODING = AudioFormat.ENCODING_PCM_16BIT;
  private static final int AUDIO_SOURCE = AudioSource.MIC;

  // A small constant valued multiplier for setting audioRecordBufferSize. This is useful
  // to reduce buffer overflows when a lot of data needs to be read at a high
  // sample rate from the audio stream. Note that it is desirable to keep this
  // multiplier small, because very large buffer sizes can slow down blocking
  // calls to AudioRecord.read(...) when the sample rate is low for instance.
  private static final int BUFFER_SIZE_MULTIPLIER = 2;

  // Number of microseconds of data to be read before sending audio data to a client. Smaller values
  // for this constant favor faster blocking calls to readAudioPacket(...).
  private static final long DEFAULT_READ_INTERVAL_MICROS = 10_000;

  // This class uses AudioFormat.ENCODING_PCM_16BIT, i.e. 16 bits per sample.
  private static final int BYTES_PER_SAMPLE = 2;

  private static final long UNINITIALIZED_TIMESTAMP = Long.MIN_VALUE;
  private static final long NANOS_PER_MICROS = 1_000;
  private static final long MICROS_PER_SECOND = 1_000_000;
  private static final long NANOS_PER_SECOND = 1_000_000_000;

  // Number of audio samples recorded per second.
  private final int sampleRateInHz;
  // Channel configuration of audio source, one of AudioRecord.CHANNEL_IN_MONO or
  // AudioRecord.CHANNEL_IN_STEREO.
  private final int channelConfig;
  // Bytes per audio frame. A frame is defined as a multi-channel audio sample. Possible values are
  // 2 bytes for 1 channel, or 4 bytes for 2 channel audio.
  private final int bytesPerFrame;
  // The minimum buffer size required by AudioRecord.
  private final int minBufferSize;

  // Number of microseconds of data to be read before sending audio data to a client. This is
  // initialized to DEFAULT_READ_INTERVAL_MICROS but can be changed by the client before calling
  // startMicrophone(...).
  private long readIntervalMicros = DEFAULT_READ_INTERVAL_MICROS;
  // Data storage allocated to internal buffer used by AudioRecord for reading audio data.
  private int audioRecordBufferSize;
  // Size of audio packet sent to an AudioConsumer with every call to consumer.onNewAudioData(...).
  private int audioPacketBufferSize;

  // Initial timestamp base. Can be set by the client so that all timestamps calculated using the
  // number of samples read per AudioRecord.read() function call start from this timestamp.
  private long initialTimestampNanos = UNINITIALIZED_TIMESTAMP;
  // The timestamp marked when startMicrophone(...) call starts recording.
  private long startRecordingTimestampNanos = UNINITIALIZED_TIMESTAMP;

  // AudioRecord is used to setup a way to record data from the audio source. See
  // https://developer.android.com/reference/android/media/AudioRecord.htm for details.
  private AudioRecord audioRecord;
  private AudioFormat audioFormat;
  // Data is read on a separate non-blocking thread.
  private Thread recordingThread;

  // This flag determines if audio will be read from the audio source and if the data read will be
  // sent to the listener of this class.
  private boolean recording = false;

  // If true, the class will drop non-increasing timestamps.
  private boolean dropNonIncreasingTimestamps;
  // Keeps track of the timestamp to guarantee that timestamps produced are always monotonically
  // increasing.
  private long lastTimestampMicros = UNINITIALIZED_TIMESTAMP;

  // The consumers are provided with the data read on every AudioRecord.read() call. If the consumer
  // called stopMicrophone() while a call to AudioRecord.read() was blocked, the class will discard
  // the data read after recording stopped.
  private final CopyOnWriteArraySet<AudioDataConsumer> consumers = new CopyOnWriteArraySet<>();

  // TODO: Add a constructor that takes an AudioFormat.

  /**
   * MicrophoneHelper class constructor. Arugments:
   *
   * @param sampleRateInHz Number of samples per second to be read from audio stream.
   * @param channelConfig Configuration of audio channels. See
   *     https://developer.android.com/reference/android/media/AudioRecord.html#public-constructors_1.
   */
  public MicrophoneHelper(int sampleRateInHz, int channelConfig) {
    this.sampleRateInHz = sampleRateInHz;
    this.channelConfig = channelConfig;
    // Number of channels of audio source, depending on channelConfig.
    final int numChannels = channelConfig == AudioFormat.CHANNEL_IN_STEREO ? 2 : 1;

    bytesPerFrame = BYTES_PER_SAMPLE * numChannels;

    // The minimum buffer size required by AudioRecord.
    minBufferSize =
        AudioRecord.getMinBufferSize(
            sampleRateInHz, channelConfig, /*audioFormat=*/ AUDIO_ENCODING);

    updateBufferSizes(readIntervalMicros);
  }

  /** Sets whether to drop non-increasing timestamps. */
  public void setDropNonIncreasingTimestamps(boolean dropNonIncreasingTimestamps) {
    this.dropNonIncreasingTimestamps = dropNonIncreasingTimestamps;
  }

  /**
   * Sets readIntervalMicros. This should be set before calling {@link #startMicrophone()}.
   *
   * @param micros the number of microseconds of data MicrophoneHelper should read before calling
   *     consumer.onNewAudioData(...).
   */
  public void setReadIntervalMicros(long micros) {
    readIntervalMicros = micros;
    updateBufferSizes(readIntervalMicros);
  }

  /**
   * Updates audioPacketBufferSize and audioRecordBufferSize.
   *
   * @param micros The interval size in microseconds of the amount of audio data to be read.
   */
  private void updateBufferSizes(long micros) {
    audioPacketBufferSize =
        (int) Math.ceil(1.0 * bytesPerFrame * sampleRateInHz * micros / MICROS_PER_SECOND);
    // The size of the internal buffer should be greater than the size of the audio packet read
    // and sent to the AudioDataConsumer so that AudioRecord.
    audioRecordBufferSize = max(audioPacketBufferSize, minBufferSize) * BUFFER_SIZE_MULTIPLIER;
  }

  private void setupAudioRecord() {

    Log.d(TAG, "AudioRecord(" + sampleRateInHz + ", " + audioRecordBufferSize + ")");
    audioFormat =
        new AudioFormat.Builder()
            .setEncoding(AUDIO_ENCODING)
            .setSampleRate(sampleRateInHz)
            .setChannelMask(channelConfig)
            .build();
    audioRecord =
        new AudioRecord.Builder()
            .setAudioSource(AUDIO_SOURCE)
            .setAudioFormat(audioFormat)
            .setBufferSizeInBytes(audioRecordBufferSize)
            .build();
    if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
      audioRecord.release();
      Log.e(TAG, "AudioRecord could not open.");
      return;
    }

    recordingThread =
        new Thread(
            () -> {
              android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

              startRecordingTimestampNanos = System.nanoTime();
              lastTimestampMicros = UNINITIALIZED_TIMESTAMP;
              long timestampOffsetNanos = 0;
              // The total number of frames read from multiple calls to AudioRecord.read() in this
              // recording thread.
              int totalNumFramesRead = 0;
              while (recording) {
                if (audioRecord == null) {
                  break;
                }

                // TODO: Fix audio data cloning.
                ByteBuffer audioData = ByteBuffer.allocateDirect(audioPacketBufferSize);
                try {
                  readAudioPacket(audioData);
                } catch (IOException ioException) {
                  // Reading audio data failed in this loop iteration, continue to next iteration if
                  // recording is still enabled.
                  Log.e(TAG, ioException.getMessage());
                  continue;
                }

                // Get the timestamp of the first audio frame which marks the beginning of reading
                // the audio data for the entire audioPacketBufferSize bytes to be read before
                // sending them to the AudioDataConsumer. We deliberately do this _after_ the read
                // call, even though we are getting the timestamp corresponding to the beginning of
                // the read chunk. This is because experimentation has shown that calling
                // AudioRecord.getTimestamp(...) before the first AudioRecord.read(...) call fails.
                long timestampNanos = getTimestampNanos(totalNumFramesRead);
                if (totalNumFramesRead == 0 && initialTimestampNanos != UNINITIALIZED_TIMESTAMP) {
                  timestampOffsetNanos = timestampNanos - initialTimestampNanos;
                }
                long timestampMicros = (timestampNanos - timestampOffsetNanos) / NANOS_PER_MICROS;
                if (dropNonIncreasingTimestamps && timestampMicros <= lastTimestampMicros) {
                  Log.i(
                      TAG, "Dropping mic audio with non-increasing timestamp: " + timestampMicros);
                  continue;
                }
                lastTimestampMicros = timestampMicros;

                // It is expected that audioRecord.read() will read full samples and therefore
                // number of bytes read is expected to be a multiple of bytesPerFrame.
                int numFramesRead = audioData.limit() / bytesPerFrame;
                totalNumFramesRead += numFramesRead;

                // Confirm that the consumer is still interested in receiving audio data and
                // stopMicrophone() wasn't called. If the consumer called stopMicrophone(), discard
                // the data read in the latest AudioRecord.read(...) function call.
                if (recording) {
                  for (AudioDataConsumer consumer : consumers) {
                    consumer.onNewAudioData(audioData, timestampMicros, audioFormat);
                  }
                }
              }
            },
            "microphoneHelperRecordingThread");
  }

  /**
   * Reads audio data into a packet.
   *
   * @param audioPacket the ByteBuffer in which audio data is read.
   * @throws java.io.IOException when AudioRecord.read(...) fails.
   */
  private void readAudioPacket(ByteBuffer audioPacket) throws IOException {
    int totalNumBytesRead = 0;
    while (totalNumBytesRead < audioPacket.capacity()) {
      int bytesRemaining = audioPacket.capacity() - totalNumBytesRead;
      int numBytesRead = 0;
      // Blocking reads are available in only API Level 23 and above.
      // https://developer.android.com/reference/android/media/AudioRecord.html#read(java.nio.ByteBuffer,%20int,%20int).
      // Note that this AudioRecord.read() fills the audio ByteBuffer in native order according to
      // the reference above, which matches further MediaPipe audio processing from the requirement
      // of PacketCreator.createAudioPacket() with this output ByteBuffer.
      if (VERSION.SDK_INT >= VERSION_CODES.M) {
        numBytesRead =
            audioRecord.read(
                audioPacket, /*sizeInBytes=*/ bytesRemaining, AudioRecord.READ_BLOCKING);
      } else {
        numBytesRead = audioRecord.read(audioPacket, /*sizeInBytes=*/ bytesRemaining);
      }
      if (numBytesRead <= 0) {
        String error = "ERROR";
        if (numBytesRead == AudioRecord.ERROR_INVALID_OPERATION) {
          error = "ERROR_INVALID_OPERATION";
        } else if (numBytesRead == AudioRecord.ERROR_BAD_VALUE) {
          error = "ERROR_BAD_VALUE";
        } else if (numBytesRead == AudioRecord.ERROR_DEAD_OBJECT) {
          error = "ERROR_DEAD_OBJECT";
        }
        throw new IOException("AudioRecord.read(...) failed due to " + error);
      }

      // Advance the position of the ByteBuffer for the next read.
      totalNumBytesRead += numBytesRead;
      audioPacket.position(totalNumBytesRead);
    }
    // Reset the position of the ByteBuffer for consumption.
    audioPacket.position(0);
  }

  /**
   * If AudioRecord.getTimestamp() is available and returns without error, this function returns the
   * timestamp using AudioRecord.getTimestamp(). If the function is unavailable, it returns a
   * fallback timestamp calculated using number of samples read so far and the initial
   * System.nanoTime(). Use framePosition to be the frame count before the latest
   * AudioRecord.read(...) call to get the timestamp of the first audio frame in the latest
   * AudioRecord.read(...) call.
   */
  private long getTimestampNanos(long framePosition) {
    long referenceFrame = 0;
    long referenceTimestamp = startRecordingTimestampNanos;
    AudioTimestamp audioTimestamp = getAudioRecordTimestamp();
    if (audioTimestamp != null) {
      referenceFrame = audioTimestamp.framePosition;
      referenceTimestamp = audioTimestamp.nanoTime;
    }
    // Assuming the first frame is read at 0 ns, this timestamp can be at most
    // (2**63-1) / 48000 nanoseconds for a sampleRateInHz = 48kHz.
    return referenceTimestamp
        + (framePosition - referenceFrame) * NANOS_PER_SECOND / sampleRateInHz;
  }

  private AudioTimestamp getAudioRecordTimestamp() {
    Preconditions.checkNotNull(audioRecord);
    // AudioRecord.getTimestamp is only available at API Level 24 and above.
    // https://developer.android.com/reference/android/media/AudioRecord.html#getTimestamp(android.media.AudioTimestamp,%20int).
    if (VERSION.SDK_INT >= VERSION_CODES.N) {
      AudioTimestamp audioTimestamp = new AudioTimestamp();
      int status = audioRecord.getTimestamp(audioTimestamp, AudioTimestamp.TIMEBASE_MONOTONIC);
      if (status == AudioRecord.SUCCESS) {
        return audioTimestamp;
      } else {
        Log.e(TAG, "audioRecord.getTimestamp failed with status: " + status);
      }
    }
    return null;
  }

  /*
   * Returns the buffer size of the internal buffer used by AudioRecord.
   */
  public int getAudioRecordBufferSize() {
    return audioRecordBufferSize;
  }

  /*
   * Returns the packet size of the audio packet that clients will receive.
   */
  public int getAudioPacketBufferSize() {
    return audioPacketBufferSize;
  }

  /**
   * Sets initialTimestampNanos. Overrides the use of system time as the first timestamp for audio
   * packets. Not recommended. Provided to maintain compatibility with existing usage by
   * CameraRecorder.
   *
   * @param initialTimestampNanos The timestamp to be used by the first audio packet read by this
   *     class when {@link #startMicrophone()} is called.
   */
  public void setInitialTimestampNanos(long initialTimestampNanos) {
    this.initialTimestampNanos = initialTimestampNanos;
  }

  /**
   * This method sets up a new AudioRecord object for reading audio data from the microphone. It can
   * be called multiple times to restart the recording if necessary.
   */
  public void startMicrophone() {
    if (recording) {
      return;
    }

    setupAudioRecord();
    audioRecord.startRecording();
    if (audioRecord.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING) {
      Log.e(TAG, "AudioRecord couldn't start recording.");
      audioRecord.release();
      return;
    }

    recording = true;
    recordingThread.start();

    Log.d(TAG, "AudioRecord is recording audio.");
  }

  /*
   * Stops the AudioRecord object from reading data from the microphone and releases it.
   */
  public void stopMicrophone() {
    stopMicrophoneWithoutCleanup();
    cleanup();
    Log.d(TAG, "AudioRecord stopped recording audio.");
  }

  /*
   * Stops the AudioRecord object from reading data from the microphone.
   */
  public void stopMicrophoneWithoutCleanup() {
    Preconditions.checkNotNull(audioRecord);
    if (!recording) {
      return;
    }

    recording = false;
    try {
      if (recordingThread != null) {
        recordingThread.join();
      }
    } catch (InterruptedException ie) {
      Log.e(TAG, "Exception: ", ie);
    }

    audioRecord.stop();
    if (audioRecord.getRecordingState() != AudioRecord.RECORDSTATE_STOPPED) {
      Log.e(TAG, "AudioRecord.stop() didn't run properly.");
    }
  }

  /*
   * Releases the AudioRecord object when there is no ongoing recording.
   */
  public void cleanup() {
    Preconditions.checkNotNull(audioRecord);
    if (recording) {
      return;
    }
    audioRecord.release();
  }

  /*
   * Clears all the old consumers and sets this as the new sole consumer.
   */
  @Override
  public void setAudioConsumer(AudioDataConsumer consumer) {
    consumers.clear();
    consumers.add(consumer);
  }

  public void addAudioConsumer(AudioDataConsumer consumer) {
    consumers.add(consumer);
  }

  public void removeAudioConsumer(AudioDataConsumer consumer) {
    consumers.remove(consumer);
  }

  public void removeAllAudioConsumers() {
    consumers.clear();
  }
}
