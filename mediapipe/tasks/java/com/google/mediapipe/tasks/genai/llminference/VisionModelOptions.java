package com.google.mediapipe.tasks.genai.llminference;

import com.google.auto.value.AutoValue;
import java.util.Optional;

/** Options for configuring vision modality */
@AutoValue
public abstract class VisionModelOptions {
  /** Returns the path to the vision encoder model file. */
  public abstract Optional<String> getEncoderPath();

  /** Path to the vision adapter model file. */
  public abstract Optional<String> getAdapterPath();

  /** Builder for {@link VisionModelOptions}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Sets the path to the vision encoder model file. */
    public abstract Builder setEncoderPath(String encoderPath);

    /** Sets the to the vision adapter model file. */
    public abstract Builder setAdapterPath(String adapterPath);

    abstract VisionModelOptions autoBuild();

    /** Validates and builds the {@link VisionModelOptions} instance. */
    public final VisionModelOptions build() {
      return autoBuild();
    }
  }

  /** Instantiates a new VisionModelOption builder. */
  public static Builder builder() {
    return new AutoValue_VisionModelOptions.Builder();
  }
}
