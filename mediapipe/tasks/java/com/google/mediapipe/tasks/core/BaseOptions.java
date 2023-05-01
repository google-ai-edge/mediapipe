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
      return options;
    }
  }

  abstract Optional<String> modelAssetPath();

  abstract Optional<Integer> modelAssetFileDescriptor();

  abstract Optional<ByteBuffer> modelAssetBuffer();

  abstract Delegate delegate();

  public static Builder builder() {
    return new AutoValue_BaseOptions.Builder().setDelegate(Delegate.CPU);
  }
}
