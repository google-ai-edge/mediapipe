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

package com.google.mediapipe.tasks.text.textproofreader;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.tasks.core.BaseOptionsUtils;
import java.io.Closeable;
import java.io.IOException;
import java.util.Optional;

/** Performs proofreading on text. */
public class TextProofreader implements Closeable {

  private final long handle;

  private TextProofreader(long handle) {
    this.handle = handle;
  }

  /** Holds the configuration for creating a {@link TextProofreader} instance. */
  @AutoValue
  public abstract static class TextProofreaderOptions {
    public abstract String getModelPath();

    public abstract Optional<ParcelFileDescriptor> getModelAssetFileDescriptor();

    public abstract Optional<Integer> getMaxNumTokens();

    public static Builder builder() {
      return new AutoValue_TextProofreader_TextProofreaderOptions.Builder().setModelPath("");
    }

    /** Builder for {@link TextProofreaderOptions}. */
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

      /** Builds a {@link TextProofreaderOptions} instance. */
      public abstract TextProofreaderOptions build();
    }
  }

  /**
   * Performs proofreading on the input text.
   *
   * @param text The input text to be proofread.
   * @return The proofreading result.
   */
  public TextProofreaderResult proofread(String text) {
    return nativeProofread(handle, text);
  }

  /**
   * Performs asynchronous proofreading on the input text.
   *
   * @param text The input text to be proofread.
   * @param callback The callback to receive the results.
   */
  public void proofreadStreaming(String text, ProofreaderResultCallback callback) {
    nativeProofreadStreaming(handle, text, callback);
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
   * Creates a {@link TextProofreader} instance from {@link TextProofreaderOptions}.
   *
   * @param context The Android context.
   * @param options The configuration options.
   * @return A {@link TextProofreader} instance.
   */
  public static TextProofreader createFromOptions(Context context, TextProofreaderOptions options) {
    String appVersion = BaseOptionsUtils.getAppVersion(context);
    String appId = BaseOptionsUtils.getAppId(context);

    int fd = -1;
    if (options.getModelAssetFileDescriptor().isPresent()) {
      try {
        fd = options.getModelAssetFileDescriptor().get().dup().detachFd();
      } catch (IOException e) {
        if (options.getModelPath().isEmpty()) {
          throw new IllegalArgumentException(
              "TextProofreader: Failed to read from file descriptor and model path is empty.", e);
        }
        Log.w(
            "TextProofreader",
            "Failed to read from file descriptor, falling back to model path: ",
            e);
      }
    }

    long handle =
        nativeCreateHandle(
            fd,
            options.getModelPath(),
            options.getMaxNumTokens().orElse(0),
            appId,
            appVersion,
            BaseOptionsUtils.HOST_ENVIRONMENT_ANDROID,
            BaseOptionsUtils.HOST_SYSTEM_ANDROID);
    return new TextProofreader(handle);
  }

  private static native long nativeCreateHandle(
      int modelFd,
      String modelPath,
      int maxNumTokens,
      String appId,
      String appVersion,
      int hostEnvironment,
      int hostSystem);

  private static native TextProofreaderResult nativeProofread(long handle, String text);

  private static native void nativeProofreadStreaming(
      long handle, String text, ProofreaderResultCallback callback);

  private static native void nativeClose(long handle);

  /** Interface for receiving proofreading results. */
  public interface ProofreaderResultCallback {
    /** Called when a new partial proofread result is available. */
    void onNext(TextProofreaderStreamingResult result);

    /** Called when an error occurs during proofreading. */
    void onError(Throwable throwable);

    /** Called when the proofreading is complete. */
    void onDone();
  }
}
