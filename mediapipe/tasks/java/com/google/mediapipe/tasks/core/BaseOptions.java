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

package com.google.mediapipe.tasks.core;

import com.google.auto.value.AutoValue;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.Optional;

/** Options to configure MediaPipe Tasks in general. */
@AutoValue
public abstract class BaseOptions {
  /** Builder for {@link BaseOptions}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /**
     * Sets the model path to a model asset file (a tflite model or a model asset bundle file) in
     * the Android app assets folder.
     *
     * <p>Note: when model path is set, both model file descriptor and model buffer should be empty.
     */
    public abstract Builder setModelAssetPath(String value);

    /**
     * Sets the native fd int of a model asset file (a tflite model or a model asset bundle file).
     *
     * <p>Note: when model file descriptor is set, both model path and model buffer should be empty.
     */
    public abstract Builder setModelAssetFileDescriptor(Integer value);

    /**
     * Sets either the direct {@link ByteBuffer} or the {@link MappedByteBuffer} of a model asset
     * file (a tflite model or a model asset bundle file).
     *
     * <p>Note: when model buffer is set, both model file and model file descriptor should be empty.
     */
    public abstract Builder setModelAssetBuffer(ByteBuffer value);

    /**
     * Sets device delegate to run the MediaPipe pipeline. If the delegate is not set, the default
     * delegate CPU is used.
     */
    public abstract Builder setDelegate(Delegate delegate);

    /** Options for the chosen delegate. If not set, the default delegate options is used. */
    public abstract Builder setDelegateOptions(DelegateOptions delegateOptions);

    abstract BaseOptions autoBuild();

    /**
     * Validates and builds the {@link BaseOptions} instance.
     *
     * @throws IllegalArgumentException if {@link BaseOptions} is invalid, or the provided model
     *     buffer is not a direct {@link ByteBuffer} or a {@link MappedByteBuffer}.
     */
    public final BaseOptions build() {
      BaseOptions options = autoBuild();
      int modelAssetPathPresent = options.modelAssetPath().isPresent() ? 1 : 0;
      int modelAssetFileDescriptorPresent = options.modelAssetFileDescriptor().isPresent() ? 1 : 0;
      int modelAssetBufferPresent = options.modelAssetBuffer().isPresent() ? 1 : 0;

      if (modelAssetPathPresent + modelAssetFileDescriptorPresent + modelAssetBufferPresent != 1) {
        throw new IllegalArgumentException(
            "Please specify only one of the model asset path, the model asset file descriptor, and"
                + " the model asset buffer.");
      }
      if (options.modelAssetBuffer().isPresent()
          && !(options.modelAssetBuffer().get().isDirect()
              || options.modelAssetBuffer().get() instanceof MappedByteBuffer)) {
        throw new IllegalArgumentException(
            "The model buffer should be either a direct ByteBuffer or a MappedByteBuffer.");
      }
      boolean delegateMatchesDelegateOptions = true;
      if (options.delegateOptions().isPresent()) {
        switch (options.delegate()) {
          case CPU:
            delegateMatchesDelegateOptions =
                options.delegateOptions().get() instanceof DelegateOptions.CpuOptions;
            break;
          case GPU:
            delegateMatchesDelegateOptions =
                options.delegateOptions().get() instanceof DelegateOptions.GpuOptions;
            break;
        }
        if (!delegateMatchesDelegateOptions) {
          throw new IllegalArgumentException(
              "Specified Delegate type does not match the provided delegate options.");
        }
      }
      return options;
    }
  }

  abstract Optional<String> modelAssetPath();

  abstract Optional<Integer> modelAssetFileDescriptor();

  abstract Optional<ByteBuffer> modelAssetBuffer();

  abstract Delegate delegate();

  abstract Optional<DelegateOptions> delegateOptions();

  /** Advanced config options for the used delegate. */
  public abstract static class DelegateOptions {

    /** Options for CPU. */
    @AutoValue
    public abstract static class CpuOptions extends DelegateOptions {

      public static Builder builder() {
        Builder builder = new AutoValue_BaseOptions_DelegateOptions_CpuOptions.Builder();
        return builder;
      }

      /** Builder for {@link CpuOptions}. */
      @AutoValue.Builder
      public abstract static class Builder {

        public abstract CpuOptions build();
      }
    }

    /** Options for GPU. */
    @AutoValue
    public abstract static class GpuOptions extends DelegateOptions {
      // Load pre-compiled serialized binary cache to accelerate init process.
      // Only available on Android. Kernel caching will only be enabled if this
      // path is set. NOTE: binary cache usage may be skipped if valid serialized
      // model, specified by "serialized_model_dir", exists.
      abstract Optional<String> cachedKernelPath();

      // A dir to load from and save to a pre-compiled serialized model used to
      // accelerate init process.
      // NOTE: serialized model takes precedence over binary cache
      // specified by "cached_kernel_path", which still can be used if
      // serialized model is invalid or missing.
      abstract Optional<String> serializedModelDir();

      // Unique token identifying the model. Used in conjunction with
      // "serialized_model_dir". It is the caller's responsibility to ensure
      // there is no clash of the tokens.
      abstract Optional<String> modelToken();

      public static Builder builder() {
        return new AutoValue_BaseOptions_DelegateOptions_GpuOptions.Builder();
      }

      /** Builder for {@link GpuOptions}. */
      @AutoValue.Builder
      public abstract static class Builder {
        public abstract Builder setCachedKernelPath(String cachedKernelPath);

        public abstract Builder setSerializedModelDir(String serializedModelDir);

        public abstract Builder setModelToken(String modelToken);

        public abstract GpuOptions build();
      }
    }
  }

  public static Builder builder() {
    return new AutoValue_BaseOptions.Builder().setDelegate(Delegate.CPU);
  }
}
