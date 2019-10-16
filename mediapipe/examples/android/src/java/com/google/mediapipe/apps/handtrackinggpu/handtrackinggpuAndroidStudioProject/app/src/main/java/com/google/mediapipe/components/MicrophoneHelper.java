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
import javax.annotation.Nullable;

/** Provides access to audio data from a microphone. */
public class MicrophoneHelper {
  /** The listener is called when audio data from the microphone is available. */
  public interface OnAudioDataAvailableListener {
    public void onAudioDataAvailable(byte[] audioData, long timestampMicros);
  }

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

  // This class uses AudioFormat.ENCODING_PCM_16BIT, i.e. 16 bits per single channel sample.
  private static final int BYTES_PER_MONO_SAMPLE = 2;

  private static final long UNINITIALIZED_TIMESTAMP = -1;
  private static final long NANOS_PER_MICROS = 1000;
  private static final long MICROS_PER_SECOND = 1000000;

  // Number of audio samples recorded per second.
  private final int sampleRateInHz;
  // Channel configuration of audio source, one of AudioRecord.CHANNEL_IN_MONO or
  // AudioRecord.CHANNEL_IN_STEREO.
  private final int channelConfig;
  // Data storage allocated to record audio samples in a single function call to AudioRecord.read().
  private final int bufferSize;
  // Bytes used per sample, accounts for number of channels of audio source. Possible values are 2
  // bytes for a 1-channel sample and 4 bytes for a 2-channel sample.
  private final int bytesPerSample;

  private byte[] audioData;

  // Timestamp provided by the AudioTimestamp object.
  private AudioTimestamp audioTimestamp;
  // Initial timestamp base. Can be set by the client so that all timestamps calculated using the
  // number of samples read per AudioRecord.read() function call start from this timestamp.
  private long initialTimestamp = UNINITIALIZED_TIMESTAMP;
  // The total number of samples read from multiple calls to AudioRecord.read(). This is reset to
  // zero for every startMicrophone() call.
  private long totalNumSamplesRead;

  // AudioRecord is used to setup a way to record data from the audio source. See
  // https://developer.android.com/reference/android/media/AudioRecord.htm for details.
  private AudioRecord audioRecord;
  // Data is read on a separate non-blocking thread.
  private Thread recordingThread;

  // This flag determines if audio will be read from the audio source and if the data read will be
  // sent to the listener of this class.
  private boolean recording = false;

  // This listener is provided with the data read on every AudioRecord.read() call. If the listener
  // called stopRecording() while a call to AudioRecord.read() was blocked, the class will discard
  // the data read after recording stopped.
  private OnAudioDataAvailableListener onAudioDataAvailableListener;

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
    final int channelCount = channelConfig == AudioFormat.CHANNEL_IN_STEREO ? 2 : 1;

    bytesPerSample = BYTES_PER_MONO_SAMPLE * channelCount;

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
      bufferSize = sampleRateInHz * MAX_READ_INTERVAL_SEC * bytesPerSample * BUFFER_SIZE_MULTIPLIER;
    } else {
      bufferSize = minBufferSize * BUFFER_SIZE_MULTIPLIER;
    }
  }

  private void setupAudioRecord() {
    audioData = new byte[bufferSize];

    Log.d(TAG, "AudioRecord(" + sampleRateInHz + ", " + bufferSize + ")");
    audioRecord =
        new AudioRecord.Builder()
            .setAudioSource(AUDIO_SOURCE)
            .setAudioFormat(
                new AudioFormat.Builder()
                    .setEncoding(AUDIO_ENCODING)
                    .setSampleRate(sampleRateInHz)
                    .setChannelMask(channelConfig)
                    .build())
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
              Log.v(TAG, "Running audio recording thread.");

              // Initial timestamp in case the AudioRecord.getTimestamp() function is unavailable.
              long startTimestamp = initialTimestamp != UNINITIALIZED_TIMESTAMP
                  ? initialTimestamp
                  : System.nanoTime() / NANOS_PER_MICROS;
              long sampleBasedTimestamp;
              while (recording) {
                if (audioRecord == null) {
                  break;
                }
                final int numBytesRead =
                    audioRecord.read(audioData, /*offsetInBytes=*/ 0, /*sizeInBytes=*/ bufferSize);
                // If AudioRecord.getTimestamp() is unavailable, calculate the timestamp using the
                // number of samples read in the call to AudioRecord.read().
                long sampleBasedFallbackTimestamp =
                    startTimestamp + totalNumSamplesRead * MICROS_PER_SECOND / sampleRateInHz;
                sampleBasedTimestamp =
                    getTimestamp(/*fallbackTimestamp=*/sampleBasedFallbackTimestamp);
                if (numBytesRead <= 0) {
                  if (numBytesRead == AudioRecord.ERROR_INVALID_OPERATION) {
                    Log.e(TAG, "ERROR_INVALID_OPERATION");
                  } else if (numBytesRead == AudioRecord.ERROR_BAD_VALUE) {
                    Log.e(TAG, "ERROR_BAD_VALUE");
                  }
                  continue;
                }
                Log.v(TAG, "Read " + numBytesRead + " bytes of audio data.");

                // Confirm that the listener is still interested in receiving audio data and
                // stopMicrophone() wasn't called. If the listener called stopMicrophone(), discard
                // the data read in the latest AudioRecord.read(...) function call.
                if (recording) {
                  onAudioDataAvailableListener.onAudioDataAvailable(
                      audioData.clone(), sampleBasedTimestamp);
                }

                // TODO: Replace byte[] with short[] audioData.
                // It is expected that audioRecord.read() will read full samples and therefore
                // numBytesRead is expected to be a multiple of bytesPerSample.
                int numSamplesRead = numBytesRead / bytesPerSample;
                totalNumSamplesRead += numSamplesRead;
              }
            });
  }

  // If AudioRecord.getTimestamp() is available and returns without error, this function returns the
  // timestamp using AudioRecord.getTimestamp(). If the function is unavailable, it returns a
  // fallbackTimestamp provided as an argument to this method.
  private long getTimestamp(long fallbackTimestamp) {
    // AudioRecord.getTimestamp is only available at API Level 24 and above.
    // https://developer.android.com/reference/android/media/AudioRecord.html#getTimestamp(android.media.AudioTimestamp,%20int).
    if (VERSION.SDK_INT >= VERSION_CODES.N) {
      if (audioTimestamp == null) {
        audioTimestamp = new AudioTimestamp();
      }
      int status = audioRecord.getTimestamp(audioTimestamp, AudioTimestamp.TIMEBASE_MONOTONIC);
      if (status == AudioRecord.SUCCESS) {
        return audioTimestamp.nanoTime / NANOS_PER_MICROS;
      } else {
        Log.e(TAG, "audioRecord.getTimestamp failed with status: " + status);
      }
    }
    return fallbackTimestamp;
  }

  // Returns the buffer size read by this class per AudioRecord.read() call.
  public int getBufferSize() {
    return bufferSize;
  }

  /**
   * Overrides the use of system time as the source of timestamps for audio packets. Not
   * recommended. Provided to maintain compatibility with existing usage by CameraRecorder.
   */
  public void setInitialTimestamp(long initialTimestamp) {
    this.initialTimestamp = initialTimestamp;
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
    totalNumSamplesRead = 0;
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
    if (recording) {
      return;
    }
    audioRecord.release();
  }

  public void setOnAudioDataAvailableListener(@Nullable OnAudioDataAvailableListener listener) {
    onAudioDataAvailableListener = listener;
  }
}
