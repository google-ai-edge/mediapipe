/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

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

package com.google.mediapipe.tasks.retrieval.universalembedder;

import com.google.auto.value.AutoValue;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import java.util.Optional;

/** Options for setting up a {@link UniversalEmbedder}. */
@AutoValue
public abstract class UniversalEmbedderOptions {

  /** Activation data type for model execution. */
  public enum ActivationDataType {
    FLOAT32(com.google.ai.edge.litertlm.ActivationDataType.FLOAT32),
    FLOAT16(com.google.ai.edge.litertlm.ActivationDataType.FLOAT16),
    INT16(com.google.ai.edge.litertlm.ActivationDataType.INT16),
    INT8(com.google.ai.edge.litertlm.ActivationDataType.INT8);

    private final com.google.ai.edge.litertlm.ActivationDataType litertLmType;

    ActivationDataType(com.google.ai.edge.litertlm.ActivationDataType litertLmType) {
      this.litertLmType = litertLmType;
    }

    public com.google.ai.edge.litertlm.ActivationDataType toLiteRtLmType() {
      return litertLmType;
    }
  }

  /** The {@link BaseOptions} for the universal embedder task. */
  public abstract BaseOptions baseOptions();

  /**
   * Optional hardware delegate to use specifically for the text encoder model. If unset, defaults
   * to the delegate configured in {@link #baseOptions()}.
   */
  public abstract Optional<Delegate> textDelegate();

  /**
   * Optional hardware delegate to use specifically for the vision encoder model. If unset, defaults
   * to the delegate configured in {@link #baseOptions()}.
   */
  public abstract Optional<Delegate> visionDelegate();

  /**
   * Optional hardware delegate to use specifically for the audio encoder model. If unset, defaults
   * to the delegate configured in {@link #baseOptions()}.
   */
  public abstract Optional<Delegate> audioDelegate();

  /**
   * Whether L2 normalization should be performed on the returned embeddings.
   *
   * <p>True by default.
   */
  public abstract boolean l2Normalize();

  /** Optional maximum input token sequence length for text embedding inputs. */
  public abstract Optional<Integer> maxInputLength();

  /** Optional number of vision tokens generated per image input. */
  public abstract Optional<Integer> visionTokensPerImage();

  /** Optional activation data type for model execution. */
  public abstract Optional<ActivationDataType> activationDataType();

  /** Optional directory path for storing compiled model cache artifacts. */
  public abstract Optional<String> cacheDir();

  /** Instantiates a new {@link Builder} for {@link UniversalEmbedderOptions}. */
  public static Builder builder() {
    return new AutoValue_UniversalEmbedderOptions.Builder().setL2Normalize(true);
  }

  /** Builder for {@link UniversalEmbedderOptions}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Sets the base options for the universal embedder task. */
    public abstract Builder setBaseOptions(BaseOptions value);

    /**
     * Sets the hardware delegate to use specifically for the text encoder model. If unset, defaults
     * to the delegate configured in {@link #setBaseOptions(BaseOptions)}.
     */
    public abstract Builder setTextDelegate(Delegate value);

    /**
     * Sets the hardware delegate to use specifically for the vision encoder model. If unset,
     * defaults to the delegate configured in {@link #setBaseOptions(BaseOptions)}.
     */
    public abstract Builder setVisionDelegate(Delegate value);

    /**
     * Sets the hardware delegate to use specifically for the audio encoder model. If unset,
     * defaults to the delegate configured in {@link #setBaseOptions(BaseOptions)}.
     */
    public abstract Builder setAudioDelegate(Delegate value);

    /**
     * Sets whether L2 normalization should be performed on the returned embeddings.
     *
     * <p>True by default.
     */
    public abstract Builder setL2Normalize(boolean value);

    /** Sets the maximum input token sequence length for text embedding inputs. */
    public abstract Builder setMaxInputLength(Integer value);

    /** Sets the number of vision tokens generated per image input. */
    public abstract Builder setVisionTokensPerImage(Integer value);

    /** Optional activation data type for model execution. */
    public abstract Builder setActivationDataType(ActivationDataType value);

    /** Optional directory path for storing compiled model cache artifacts. */
    public abstract Builder setCacheDir(String value);

    /** Builds a {@link UniversalEmbedderOptions} instance. */
    public abstract UniversalEmbedderOptions build();
  }
}
