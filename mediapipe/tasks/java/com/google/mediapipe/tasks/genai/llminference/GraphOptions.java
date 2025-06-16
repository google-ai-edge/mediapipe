package com.google.mediapipe.tasks.genai.llminference;

import com.google.auto.value.AutoValue;

/** Configuration for the inference graph. */
@AutoValue
public abstract class GraphOptions {

  /**
   * Returns whether to configure the graph to include the token cost calculator, which allows users
   * to only compute the cost of a prompt.
   */
  public abstract boolean includeTokenCostCalculator();

  /** Returns whether to configure the graph to include the vision modality. */
  public abstract boolean enableVisionModality();

  /** Returns whether to configure the graph to include the audio modality. */
  public abstract boolean enableAudioModality();

  /** Returns a new {@link Builder} instance. */
  public static Builder builder() {
    return new AutoValue_GraphOptions.Builder()
        .setIncludeTokenCostCalculator(true)
        .setEnableVisionModality(false)
        .setEnableAudioModality(false);
  }

  /** Builder for {@link GraphConfig}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Sets whether to configure the graph to include the token cost calculator. */
    public abstract Builder setIncludeTokenCostCalculator(boolean includeTokenCostCalculator);

    /** Sets whether to configure the graph to include the vision modality. */
    public abstract Builder setEnableVisionModality(boolean enableVisionModality);

    /** Sets whether to configure the graph to include the audio modality. */
    public abstract Builder setEnableAudioModality(boolean enableAudioModality);

    /** AutoValue generated builder method. */
    abstract GraphOptions autoBuild();

    /** Builds a new {@link GraphConfig} instance. */
    public GraphOptions build() {
      return autoBuild();
    }
  }
}
