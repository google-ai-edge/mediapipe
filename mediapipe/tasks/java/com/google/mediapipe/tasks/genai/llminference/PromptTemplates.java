package com.google.mediapipe.tasks.genai.llminference;

import com.google.auto.value.AutoValue;

/** Configuration for the prompt templates. */
@AutoValue
public abstract class PromptTemplates {

  /** The prefix to prepend to the user query. */
  public abstract String userPrefix();

  /** The suffix to append to the user query. */
  public abstract String userSuffix();

  /** The prefix to prepend to the model response. */
  public abstract String modelPrefix();

  /** The suffix to append to the model response. */
  public abstract String modelSuffix();

  /** The prefix to prepend to the system instructions. */
  public abstract String systemPrefix();

  /** The suffix to append to the system instructions. */
  public abstract String systemSuffix();

  /** Returns a new {@link Builder} instance. */
  public static Builder builder() {
    return new AutoValue_PromptTemplates.Builder()
        .setUserPrefix("")
        .setUserSuffix("")
        .setModelPrefix("")
        .setModelSuffix("")
        .setSystemPrefix("")
        .setSystemSuffix("");
  }

  /** Builder for {@link PromptTemplates}. */
  @AutoValue.Builder
  public abstract static class Builder {

    /** Sets the prefix to prepend to the user query. */
    public abstract Builder setUserPrefix(String userPrefix);

    /** Sets the suffix to append to the user query. */
    public abstract Builder setUserSuffix(String userSuffix);

    /** Sets the prefix to prepend to the model response. */
    public abstract Builder setModelPrefix(String modelPrefix);

    /** Sets the suffix to append to the model response. */
    public abstract Builder setModelSuffix(String modelSuffix);

    /** Sets the prefix to prepend to the system instructions. */
    public abstract Builder setSystemPrefix(String systemPrefix);

    /** Sets the suffix to append to the system instructions. */
    public abstract Builder setSystemSuffix(String systemSuffix);

    /** AutoValue generated builder method. */
    abstract PromptTemplates autoBuild();

    /** Builds a new {@link PromptTemplates} instance. */
    public PromptTemplates build() {
      return autoBuild();
    }
  }
}
