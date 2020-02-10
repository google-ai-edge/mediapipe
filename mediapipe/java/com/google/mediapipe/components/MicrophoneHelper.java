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
import android.media.AudioRecord;
import android.media.AudioTimestamp;
import android.media.MediaRecorder.AudioSource;
import android.os.Build.VERSION;
import android.os.Build.VERSION_CODES;
import android.util.Log;
import com.google.common.base.Preconditions;
import java.nio.ByteBuffer;

/** Provides access to audio data from a microphone. */
public class MicrophoneHelper implements AudioDataProducer {
  private static final String TAG = "MicrophoneHelper";

  private static final int AUDIO_ENCODING = AudioFormat.ENCODING_PCM_16BIT;
  private static final int AUDIO_SOURCE = AudioSource.MIC;

  // A small constant valued multiplier for setting bufferSize. This is useful
  // to reduce buffer overflows when a lot of data needs to be read at a high
  // sample rate from the audio stream. Note that it is desirable to keep this
  // multiplier small, because very large buffer sizes can slow down blocking
  // calls to AudioRecord.read(...) when the sample rate is low for instance.
  private static final int BUFFER_SIZE_MULTIPLIER = 2;

  // A small constant value to decide the number of seconds of audio data that
  // will be read in a single AudioRecord.read(...) call when
  // AudioRecord.minBufferSize(...) is unavailable. Smaller values for this
  // constant favor faster blocking calls to AudioRecord.read(...).
  private static final int MAX_READ_INTERVAL_SEC = 1;

  // This class uses AudioFormat.ENCODING_PCM_16BIT, i.e. 16 bits per sample.
  private static final int BYTES_PER_SAMPLE = 2;

  private static final long UNINITIALIZED_TIMESTAMP = Long.MIN_VALUE;
  private static final long NANOS_PER_MICROS = 1000;
  private static final long MICROS_PER_SECOND = 1000000;

  // Number of audio samples recorded per second.
  private final int sampleRateInHz;
  // Channel configuration of audio source, one of AudioRecord.CHANNEL_IN_MONO or
  // AudioRecord.CHANNEL_IN_STEREO.
  private final int channelConfig;
  // Bytes per audio frame. A frame is defined as a multi-channel audio sample. Possible values are
  // 2 bytes for 1 channel, or 4 bytes for 2 channel audio.
  private final int bytesPerFrame;
  // Data storage allocated to record audio samples in a single function call to AudioRecord.read().
  private final int bufferSize;

  // Initial timestamp base. Can be set by the client so that all timestamps calculated using the
  // number of samples read per AudioRecord.read() function call start from this timestamp. If it
  // is not set by the client, then every startMicrophone(...) call marks a value for it.
  private long initialTimestampMicros = UNINITIALIZED_TIMESTAMP;

  // AudioRecord is used to setup a way to record data from the audio source. See
  // https://developer.android.com/reference/android/media/AudioRecord.htm for details.
  private AudioRecord audioRecord;
  private AudioFormat audioFormat;
  // Data is read on a separate non-blocking thread.
  private Thread recordingThread;

  // This flag determines if audio will be read from the audio source and if the data read will be
  // sent to the listener of this class.
  private boolean recording = false;

  // The consumer is provided with the data read on every AudioRecord.read() call. If the consumer
  // called stopRecording() while a call to AudioRecord.read() was blocked, the class will discard
  // the data read after recording stopped.
  private AudioDataConsumer consumer;

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
    final int minBufferSize =
        AudioRecord.getMinBufferSize(
            sampleRateInHz, channelConfig, /*audioFormat=*/ AUDIO_ENCODING);

    // Set bufferSize. If the minimum buffer size permitted by the hardware is
    // unavailable, use the the sampleRateInHz value as the number of bytes.
    // This is arguably better than another arbitrary constant because a higher
    // value of sampleRateInHz implies the need for reading large chunks of data
    // from the audio stream in each AudioRecord.read(...) call.
    if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
      Log.e(TAG, "AudioRecord minBufferSize unavailable.");
      bufferSize = sampleRateInHz * MAX_READ_INTERVAL_SEC * bytesPerFrame * BUFFER_SIZE_MULTIPLIER;
    } else {
      bufferSize = minBufferSize * BUFFER_SIZE_MULTIPLIER;
    }
  }

  private void setupAudioRecord() {

    Log.d(TAG, "AudioRecord(" + sampleRateInHz + ", " + bufferSize + ")");
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
            .setBufferSizeInBytes(bufferSize)
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

              // The total number of frames read from multiple calls to AudioRecord.read() in this
              // recording thread.
              int totalNumFramesRead = 0;
              while (recording) {
                if (audioRecord == null) {
                  break;
                }
                // TODO: Fix audio data cloning.
                ByteBuffer audioData = ByteBuffer.allocateDirect(bufferSize);
                final int numBytesRead = audioRecord.read(audioData, /*sizeInBytes=*/ bufferSize);
                // Get the timestamp of the first audio frame in the latest read call.
                long timestampMicros = getTimestampMicros(totalNumFramesRead);
                if (numBytesRead <= 0) {
                  if (numBytesRead == AudioRecord.ERROR_INVALID_OPERATION) {
                    Log.e(TAG, "ERROR_INVALID_OPERATION");
                  } else if (numBytesRead == AudioRecord.ERROR_BAD_VALUE) {
                    Log.e(TAG, "ERROR_BAD_VALUE");
                  }
                  continue;
                }

                // Confirm that the consumer is still interested in receiving audio data and
                // stopMicrophone() wasn't called. If the consumer called stopMicrophone(), discard
                // the data read in the latest AudioRecord.read(...) function call.
                if (recording && consumer != null) {
                  consumer.onNewAudioData(audioData, timestampMicros, audioFormat);
                }

                // It is expected that audioRecord.read() will read full samples and therefore
                // numBytesRead is expected to be a multiple of bytesPerFrame.
                int numFramesRead = numBytesRead / bytesPerFrame;
                totalNumFramesRead += numFramesRead;
              }
            },
            "microphoneHelperRecordingThread");
  }

  // If AudioRecord.getTimestamp() is available and returns without error, this function returns the
  // timestamp using AudioRecord.getTimestamp(). If the function is unavailable, it returns a
  // fallback timestamp calculated using number of samples read so far.
  // Use numFramesRead to be the frame count before the latest AudioRecord.read(...) call to get
  // the timestamp of the first audio frame in the latest AudioRecord.read(...) call.
  private long getTimestampMicros(long numFramesRead) {
    AudioTimestamp audioTimestamp = getAudioRecordTimestamp();
    if (audioTimestamp == null) {
      if (numFramesRead == 0) {
        initialTimestampMicros = markInitialTimestamp();
      }

      // If AudioRecord.getTimestamp() is unavailable, calculate the timestamp using the
      // number of frames read in the call to AudioRecord.read().
      return initialTimestampMicros + numFramesRead * getMicrosPerSample();
    }
    // If audioTimestamp.framePosition is ahead of numFramesRead so far, then the offset is
    // negative.
    long frameOffset = numFramesRead - audioTimestamp.framePosition;
    long audioTsMicros = audioTimestamp.nanoTime / NANOS_PER_MICROS;
    return audioTsMicros + frameOffset * getMicrosPerSample();
  }

  private long markInitialTimestamp() {
    return initialTimestampMicros != UNINITIALIZED_TIMESTAMP
        ? initialTimestampMicros
        : System.nanoTime() / NANOS_PER_MICROS;
  }

  private long getMicrosPerSample() {
    return MICROS_PER_SECOND / sampleRateInHz;
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

  // Returns the buffer size read by this class per AudioRecord.read() call.
  public int getBufferSize() {
    return bufferSize;
  }

  /**
   * Overrides the use of system time as the source of timestamps for audio packets. Not
   * recommended. Provided to maintain compatibility with existing usage by CameraRecorder.
   */
  public void setInitialTimestampMicros(long initialTimestampMicros) {
    this.initialTimestampMicros = initialTimestampMicros;
  }

  // This method sets up a new AudioRecord object for reading audio data from the microphone. It
  // can be called multiple times to restart the recording if necessary.
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

  // Stops the AudioRecord object from reading data from the microphone and releases it.
  public void stopMicrophone() {
    stopMicrophoneWithoutCleanup();
    cleanup();
    Log.d(TAG, "AudioRecord stopped recording audio.");
  }

  // Stops the AudioRecord object from reading data from the microphone.
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

  // Releases the AudioRecord object when there is no ongoing recording.
  public void cleanup() {
    Preconditions.checkNotNull(audioRecord);
    if (recording) {
      return;
    }
    audioRecord.release();
  }

  @Override
  public void setAudioConsumer(AudioDataConsumer consumer) {
    this.consumer = consumer;
  }
}
