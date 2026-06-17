/* Copyright 2026 The MediaPipe Authors.

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

package com.google.mediapipe.tasks.text.textsummarizer;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.tasks.core.BaseOptionsUtils;
import java.io.Closeable;
import java.io.IOException;
import java.util.Optional;

/** Performs summarization on text. */
public class TextSummarizer implements Closeable {

  private final long handle;

  private TextSummarizer(long handle) {
    this.handle = handle;
  }

  /** Holds the configuration for creating a {@link TextSummarizer} instance. */
  @AutoValue
  public abstract static class TextSummarizerOptions {
    public abstract String getModelPath();

    public abstract Optional<ParcelFileDescriptor> getModelAssetFileDescriptor();

    public abstract Optional<Integer> getMaxNumTokens();

    public abstract Mode getMode();

    public static Builder builder() {
      return new AutoValue_TextSummarizer_TextSummarizerOptions.Builder()
          .setModelPath("")
          .setMode(Mode.KEYPOINTS);
    }

    /** The mode of the text summarizer. */
    public enum Mode {
      TLDR(0),
      KEYPOINTS(1);

      private final int value;

      Mode(int value) {
        this.value = value;
      }

      public int getValue() {
        return value;
      }
    }

    /** Builder for {@link TextSummarizerOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the local path or identifier for the model files. */
      public abstract Builder setModelPath(String modelPath);

      /** Sets the file descriptor of the model. */
      public abstract Builder setModelAssetFileDescriptor(
          ParcelFileDescriptor modelAssetFileDescriptor);

      /**
       * Sets the maximum context length (in tokens) to use. If not set, the model's default maximum
       * context length will be used. If the value set is larger than the model's maximum context
       * length, the model's maximum context length is used.
       */
      public abstract Builder setMaxNumTokens(Integer maxNumTokens);

      /** Sets the mode of the text summarizer. Defaults to [Mode.KEYPOINTS]. */
      public abstract Builder setMode(Mode mode);

      /** Builds a {@link TextSummarizerOptions} instance. */
      public abstract TextSummarizerOptions build();
    }
  }

  /**
   * Performs summarization on the input text.
   *
   * @param text The input text to be summarized.
   * @return The summarization result.
   */
  public TextSummarizerResult summarize(String text) {
    return nativeSummarize(handle, text);
  }

  /**
   * Performs summarization on the input text with a callback.
   *
   * @param text The input text to be summarized.
   * @param callback The callback to receive the results.
   */
  public void summarizeStreaming(String text, SummarizationResultCallback callback) {
    nativeSummarizeStreaming(handle, text, callback);
  }

  @Override
  public void close() {
    nativeClose(handle);
  }

  static {
    try {
      System.loadLibrary("mediapipe_tasks_textgenai_jni");
    } catch (UnsatisfiedLinkError e) {
      // Library might have been loaded manually (e.g. in tests)
      // or unavailable. If unavailable, native methods will fail later.
      System.err.println("Failed to load mediapipe_tasks_textgenai_jni: " + e.getMessage());
    }
  }

  /**
   * Creates a {@link TextSummarizer} instance from {@link TextSummarizerOptions}.
   *
   * @param context The Android context.
   * @param options The configuration options.
   * @return A {@link TextSummarizer} instance.
   */
  public static TextSummarizer createFromOptions(Context context, TextSummarizerOptions options) {
    String appVersion = BaseOptionsUtils.getAppVersion(context);
    String appId = BaseOptionsUtils.getAppId(context);

    int fd = -1;
    if (options.getModelAssetFileDescriptor().isPresent()) {
      try {
        fd = options.getModelAssetFileDescriptor().get().dup().detachFd();
      } catch (IOException e) {
        if (options.getModelPath().isEmpty()) {
          throw new IllegalArgumentException(
              "TextSummarizer: Failed to read from file descriptor and model path is empty.", e);
        }
        Log.w(
            "TextSummarizer",
            "Failed to read from file descriptor, falling back to model path: ",
            e);
      }
    }

    long handle =
        nativeCreateHandle(
            fd,
            options.getModelPath(),
            options.getMaxNumTokens().orElse(0),
            options.getMode().getValue(),
            appId,
            appVersion,
            BaseOptionsUtils.HOST_ENVIRONMENT_ANDROID,
            BaseOptionsUtils.HOST_SYSTEM_ANDROID);
    return new TextSummarizer(handle);
  }

  private static native long nativeCreateHandle(
      int modelFd,
      String modelPath,
      int maxNumTokens,
      int mode,
      String appId,
      String appVersion,
      int hostEnvironment,
      int hostSystem);

  private static native TextSummarizerResult nativeSummarize(long handle, String text);

  private static native void nativeSummarizeStreaming(
      long handle, String text, SummarizationResultCallback callback);

  private static native void nativeClose(long handle);

  /** Interface for receiving summarization results. */
  public interface SummarizationResultCallback {
    /** Called when a new partial summary is available. */
    void onNext(TextSummarizerStreamingResult result);

    /** Called when an error occurs during summarization. */
    void onError(Throwable throwable);

    /** Called when the summarization is complete. */
    void onDone();
  }
}
