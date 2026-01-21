package com.google.mediapipe.tasks.genai.llminference;

import com.google.auto.value.AutoValue;

/** Options for configuring the audio modality. */
@AutoValue
public abstract class AudioModelOptions {

  /** Returns the maximum audio sequence length for the audio encoder. */
  public abstract int maxAudioSequenceLength();

  /** Returns a new {@link Builder} instance. */
  public static Builder builder() {
    return new AutoValue_AudioModelOptions.Builder().setMaxAudioSequenceLength(0);
  }

  /** Builder for {@link AudioModelOptions}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Sets the maximum audio sequence length for the audio encoder. */
    public abstract Builder setMaxAudioSequenceLength(int maxAudioSequenceLength);

    /** AutoValue generated builder method. */
    abstract AudioModelOptions autoBuild();

    /** Builds a new {@link AudioModelOptions} instance. */
    public AudioModelOptions build() {
      return autoBuild();
    }
  }
}
