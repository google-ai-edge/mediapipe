// Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

package com.google.mediapipe.tasks.components.processors;

import com.google.auto.value.AutoValue;
import com.google.mediapipe.tasks.components.processors.proto.EmbedderOptionsProto;

/** Embedder options shared across MediaPipe Java embedding tasks. */
@AutoValue
public abstract class EmbedderOptions {

  /** Builder for {@link EmbedderOptions} */
  @AutoValue.Builder
  public abstract static class Builder {
    /**
     * Sets whether L2 normalization should be performed on the returned embeddings. Use this option
     * only if the model does not already contain a native <code>L2_NORMALIZATION</code> TF Lite Op.
     * In most cases, this is already the case and L2 norm is thus achieved through TF Lite
     * inference.
     *
     * <p>False by default.
     */
    public abstract Builder setL2Normalize(boolean l2Normalize);

    /**
     * Sets whether the returned embedding should be quantized to bytes via scalar quantization.
     * Embeddings are implicitly assumed to be unit-norm and therefore any dimensions is guaranteed
     * to have value in <code>[-1.0, 1.0]</code>. Use {@link #setL2Normalize(boolean)} if this is
     * not the case.
     *
     * <p>False by default.
     */
    public abstract Builder setQuantize(boolean quantize);

    public abstract EmbedderOptions build();
  }

  public abstract boolean l2Normalize();

  public abstract boolean quantize();

  public static Builder builder() {
    return new AutoValue_EmbedderOptions.Builder().setL2Normalize(false).setQuantize(false);
  }

  /**
   * Converts an {@link EmbedderOptions} object to an {@link EmbedderOptionsProto.EmbedderOptions}
   * protobuf message.
   */
  public EmbedderOptionsProto.EmbedderOptions convertToProto() {
    return EmbedderOptionsProto.EmbedderOptions.newBuilder()
        .setL2Normalize(l2Normalize())
        .setQuantize(quantize())
        .build();
  }
}
