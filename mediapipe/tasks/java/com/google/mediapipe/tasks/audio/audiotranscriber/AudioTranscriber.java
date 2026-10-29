// Copyright 2026 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.audio.audiotranscriber;

import android.content.Context;
import androidx.annotation.WorkerThread;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.BaseOptionsUtils;
import com.google.mediapipe.tasks.core.TaskOptions;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import java.util.Optional;

/** Performs streaming and non-streaming speech-to-text transcription on mono PCM float audio. */
public final class AudioTranscriber implements AutoCloseable {
  private static final int DEFAULT_SAMPLE_RATE_HZ = 16000;
  private static final int DELEGATE_CPU = 1;
  private static final int DELEGATE_GPU = 2;
  private static final int DELEGATE_LITERT_NPU = 5;

  private final long nativeHandle;

  private boolean isClosed = false;

  static {
    try {
      System.loadLibrary("mediapipe_tasks_audio_jni");
    } catch (UnsatisfiedLinkError e) {
      // Library might have been loaded manually (e.g. in tests)
      // or unavailable. If unavailable, native methods will fail later.
      System.err.println("Failed to load mediapipe_tasks_audio_jni: " + e.getMessage());
    }
  }

  private AudioTranscriber(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  /**
   * Creates an {@link AudioTranscriber} instance from a model file and the default {@link
   * AudioTranscriberOptions}.
   *
   * @param context The Android {@link Context}.
   * @param modelPath Path to the transcription model file.
   * @return An {@link AudioTranscriber} instance.
   */
  public static AudioTranscriber createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, AudioTranscriberOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates an {@link AudioTranscriber} instance from {@link AudioTranscriberOptions}.
   *
   * @param context The Android {@link Context}.
   * @param options The {@link AudioTranscriberOptions} instance.
   * @return An {@link AudioTranscriber} instance.
   */
  public static AudioTranscriber createFromOptions(
      Context context, AudioTranscriberOptions options) {
    BaseOptionsProto.BaseOptions baseOptionsProto =
        BaseOptionsUtils.convertBaseOptionsToProto(options.baseOptions());
    if (baseOptionsProto.getModelAsset().hasFileContent()
        || baseOptionsProto.getModelAsset().hasFilePointerMeta()) {
      throw new IllegalArgumentException(
          "AudioTranscriber does not support in-memory model buffers; provide a model path or"
              + " file descriptor instead.");
    }
    String modelPath =
        baseOptionsProto.getModelAsset().getFileName().isEmpty()
            ? null
            : baseOptionsProto.getModelAsset().getFileName();

    int fileDescriptor =
        baseOptionsProto.getModelAsset().hasFileDescriptorMeta()
            ? baseOptionsProto.getModelAsset().getFileDescriptorMeta().getFd()
            : -1;
    int delegate = toNativeDelegate(baseOptionsProto);
    String appId = BaseOptionsUtils.getAppId(context);
    String appVersion = BaseOptionsUtils.getAppVersion(context);

    long handle =
        nativeCreateAudioTranscriber(
            modelPath,
            fileDescriptor,
            options.sampleRateHz(),
            options.enableInterimResults(),
            delegate,
            options.modelArchitecture().value(),
            options.prompt().orElse(null),
            options.cacheDir().orElse(null),
            appId,
            appVersion);
    return new AudioTranscriber(handle);
  }

  /**
   * Transcribes a non-terminal chunk of mono PCM float audio samples in {@code [-1.0f, 1.0f]}.
   *
   * @param audioSamples Mono PCM float audio samples.
   * @return The {@link AudioTranscriberResult} for this chunk.
   */
  @WorkerThread
  public AudioTranscriberResult transcribe(float[] audioSamples) {
    return transcribe(audioSamples, /* endOfStream= */ false);
  }

  /**
   * Transcribes a chunk of mono PCM float audio samples in {@code [-1.0f, 1.0f]}.
   *
   * @param audioSamples Mono PCM float audio samples.
   * @param endOfStream Whether this chunk marks the end of the audio stream.
   * @return The {@link AudioTranscriberResult} for this chunk.
   */
  @WorkerThread
  @SuppressWarnings("PreferPreconditions") // 3P API that does not depend on Guava
  public AudioTranscriberResult transcribe(float[] audioSamples, boolean endOfStream) {
    checkNotClosed();
    if (audioSamples == null) {
      throw new IllegalArgumentException("audioSamples must not be null.");
    }
    return nativeTranscribe(nativeHandle, audioSamples, endOfStream);
  }

  /**
   * Flushes any remaining buffered audio and finalizes the current stream.
   *
   * @return The final {@link AudioTranscriberResult}.
   */
  @WorkerThread
  public AudioTranscriberResult finishStream() {
    checkNotClosed();
    return nativeFinishStream(nativeHandle);
  }

  /** Prewarms the underlying inference engine to reduce first-chunk latency. */
  @WorkerThread
  public void prewarm() {
    checkNotClosed();
    nativePrewarm(nativeHandle);
  }

  @Override
  public void close() {
    if (isClosed) {
      return;
    }
    isClosed = true;
    nativeCloseAudioTranscriber(nativeHandle);
  }

  private void checkNotClosed() {
    if (isClosed) {
      throw new IllegalStateException("AudioTranscriber has already been closed.");
    }
  }

  private static int toNativeDelegate(BaseOptionsProto.BaseOptions proto) {
    if (proto.getAcceleration().hasGpu()) {
      return DELEGATE_GPU;
    }
    if (proto.getAcceleration().hasLitert()) {
      if (proto.getAcceleration().getLitert().hasNpu()) {
        return DELEGATE_LITERT_NPU;
      }
      if (proto.getAcceleration().getLitert().hasGpu()) {
        return DELEGATE_GPU;
      }
    }
    return DELEGATE_CPU;
  }

  /** Model architecture backend family for {@link AudioTranscriber}. */
  public enum ModelArchitecture {
    /** Automatically infer the model architecture from the model asset. */
    AUTO(0),
    /** Open-weight ASR models executed via the LiteRT Omni ASR runtime. */
    OMNI(1),
  ;

    private final int value;

    ModelArchitecture(int value) {
      this.value = value;
    }

    int value() {
      return value;
    }
  }

  /** Options for setting up an {@link AudioTranscriber}. */
  @AutoValue
  public abstract static class AudioTranscriberOptions extends TaskOptions {

    /** Builder for {@link AudioTranscriberOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the {@link BaseOptions} for the audio transcriber task. */
      public abstract Builder setBaseOptions(BaseOptions baseOptions);

      /** Sets the input audio sample rate in Hz (defaults to 16000 Hz). */
      public abstract Builder setSampleRateHz(int sampleRateHz);

      /** Sets whether to emit interim (non-final) transcription segments during streaming. */
      public abstract Builder setEnableInterimResults(boolean enableInterimResults);

      /** Sets the model architecture family (defaults to {@link ModelArchitecture#AUTO}). */
      public abstract Builder setModelArchitecture(ModelArchitecture modelArchitecture);

      /** Sets the optional system/user prompt override for generative ASR models. */
      public abstract Builder setPrompt(String prompt);

      /** Sets the optional directory path for caching compiled model artifacts. */
      public abstract Builder setCacheDir(String cacheDir);

      abstract AudioTranscriberOptions autoBuild();

      /** Builds the {@link AudioTranscriberOptions} instance. */
      @SuppressWarnings("PreferPreconditions") // 3P API that does not depend on Guava
      public final AudioTranscriberOptions build() {
        AudioTranscriberOptions options = autoBuild();
        if (options.sampleRateHz() <= 0) {
          throw new IllegalArgumentException(
              "sampleRateHz must be positive, got: " + options.sampleRateHz());
        }
        return options;
      }
    }

    public abstract BaseOptions baseOptions();

    public abstract int sampleRateHz();

    public abstract boolean enableInterimResults();

    public abstract ModelArchitecture modelArchitecture();

    public abstract Optional<String> prompt();

    public abstract Optional<String> cacheDir();

    public static Builder builder() {
      return new AutoValue_AudioTranscriber_AudioTranscriberOptions.Builder()
          .setSampleRateHz(DEFAULT_SAMPLE_RATE_HZ)
          .setEnableInterimResults(false)
          .setModelArchitecture(ModelArchitecture.AUTO);
    }
  }

  private static native long nativeCreateAudioTranscriber(
      String modelPath,
      int fileDescriptor,
      int sampleRateHz,
      boolean enableInterimResults,
      int delegate,
      int modelArchitecture,
      String prompt,
      String cacheDir,
      String appId,
      String appVersion);

  private static native AudioTranscriberResult nativeTranscribe(
      long nativeHandle, float[] audioSamples, boolean endOfStream);

  private static native AudioTranscriberResult nativeFinishStream(long nativeHandle);

  private static native void nativePrewarm(long nativeHandle);

  private static native void nativeCloseAudioTranscriber(long nativeHandle);
}
