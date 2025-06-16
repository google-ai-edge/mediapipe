package com.google.mediapipe.tasks.genai.llminference;


import android.content.Context;
import com.google.auto.value.AutoValue;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.tasks.genai.llminference.jni.proto.LlmOptionsProto.LlmModelSettings;
import com.google.mediapipe.tasks.genai.llminference.jni.proto.LlmOptionsProto.LlmModelSettings.LlmPreferredBackend;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

/** LlmInference Task Java API */
public class LlmInference implements AutoCloseable {
  /** The backend to use for inference. */
  public enum Backend {
    /** Use the default backend for the model. */
    DEFAULT,
    /** Use the CPU backend for inference. */
    CPU,
    /** Use the GPU backend for inference. */
    GPU
  }

  private static final String STATS_TAG = LlmInference.class.getSimpleName();

  private static final int NUM_DECODE_STEPS_PER_SYNC = 3;

  private final LlmTaskRunner taskRunner;

  /**
   * An implicit session for all request that do use the public session API. These sessions are
   * short-lived and are only kept for a single inference.
   */
  private final AtomicReference<LlmInferenceSession> implicitSession;

  static {
    System.loadLibrary("llm_inference_engine_jni");
  }

  /** Creates an LlmInference Task. */
  public static LlmInference createFromOptions(Context context, LlmInferenceOptions options) {
    // Configure LLM model settings.
    LlmModelSettings.Builder modelSettings =
        LlmModelSettings.newBuilder()
            .setModelPath(options.modelPath())
            .setCacheDir(context.getCacheDir().getAbsolutePath())
            .setNumDecodeStepsPerSync(NUM_DECODE_STEPS_PER_SYNC)
            .setMaxTokens(options.maxTokens())
            .setMaxTopK(options.maxTopK())
            .setMaxNumImages(options.maxNumImages())
            .setNumberOfSupportedLoraRanks(options.supportedLoraRanks().size())
            .addAllSupportedLoraRanks(options.supportedLoraRanks());

    if (options.visionModelOptions().isPresent()) {
      VisionModelOptions visionModelOptions = options.visionModelOptions().get();

      LlmModelSettings.VisionModelSettings.Builder visionModelSettings =
          LlmModelSettings.VisionModelSettings.newBuilder();
      visionModelOptions.getEncoderPath().ifPresent(visionModelSettings::setEncoderPath);
      visionModelOptions.getAdapterPath().ifPresent(visionModelSettings::setAdapterPath);

      modelSettings.setVisionModelSettings(visionModelSettings.build());
    }

    if (options.audioModelOptions().isPresent()) {
      AudioModelOptions audioModelOptions = options.audioModelOptions().get();
      LlmModelSettings.AudioModelSettings.Builder audioModelSettings =
          LlmModelSettings.AudioModelSettings.newBuilder()
              .setMaxAudioSequenceLength(audioModelOptions.maxAudioSequenceLength());
      modelSettings.setAudioModelSettings(audioModelSettings.build());
    }

    if (options.preferredBackend().isPresent()) {
      switch (options.preferredBackend().get()) {
        case DEFAULT:
          modelSettings.setLlmPreferredBackend(LlmPreferredBackend.DEFAULT);
          break;
        case CPU:
          modelSettings.setLlmPreferredBackend(LlmPreferredBackend.CPU);
          break;
        case GPU:
          modelSettings.setLlmPreferredBackend(LlmPreferredBackend.GPU);
          break;
      }
    }

    return new LlmInference(context, STATS_TAG, modelSettings.build());
  }

  /** Constructor to initialize an {@link LlmInference}. */
  private LlmInference(Context context, String taskName, LlmModelSettings modelSettings) {
    this.taskRunner = new LlmTaskRunner(context, taskName, modelSettings);
    this.implicitSession = new AtomicReference<>();
  }

  /**
   * Generates a response based on the input text. This method cannot be called while other queries
   * are active.
   *
   * <p>This function creates a new session for each call. If you want to have a stateful inference,
   * use {@link LlmInferenceSession#generateResponse()} instead.
   *
   * <p>Note: You cannot invoke simultaneous response generation calls on active sessions created
   * using the same {@link LlmInference}. You have to wait for the currently running response
   * generation call to complete before initiating another one.
   *
   * @param inputText a {@link String} for processing.
   * @throws IllegalStateException if the inference fails.
   */
  public String generateResponse(String inputText) {
    LlmInferenceSession session = resetImplicitSession();
    session.addQueryChunk(inputText);
    return session.generateResponse();
  }

  /**
   * Asynchronously generates a response based on the input text. This method cannot be called while
   * other queries are active.
   *
   * <p>This function creates a new session for each call and returns the complete response as a
   * {@link ListenableFuture}. If you want to have a stateful inference, use {@link
   * LlmInferenceSession#generateResponseAsync()} instead.
   *
   * <p>Note: You cannot invoke simultaneous response generation calls on active sessions created
   * using the same {@link LlmInference}. You have to wait for the currently running response
   * generation call to complete before initiating another one.
   *
   * @param inputText a {@link String} for processing.
   * @return a {@link ListenableFuture} with the complete response once the inference is complete.
   * @throws IllegalStateException if the inference fails.
   */
  public ListenableFuture<String> generateResponseAsync(String inputText) {
    LlmInferenceSession session = resetImplicitSession();
    session.addQueryChunk(inputText);
    return session.generateResponseAsync();
  }

  /**
   * Asynchronously generates a response based on the input text and emits partial results. This
   * method cannot be called while other queries are active.
   *
   * <p>This function creates a new session for each call and returns the complete response as a
   * {@link ListenableFuture} and invokes the {@code progressListener} as the response is generated.
   * If you want to have a stateful inference, use {@link
   * LlmInferenceSession#generateResponseAsync()} instead.
   *
   * <p>Note: You cannot invoke simultaneous response generation calls on active sessions created
   * using the same {@link LlmInference}. You have to wait for the currently running response
   * generation call to complete before initiating another one.
   *
   * @param inputText a {@link String} for processing.
   * @param progressListener a {@link ProgressListener} to receive partial results.
   * @return a {@link ListenableFuture} with the complete response once the inference is complete.
   * @throws IllegalStateException if the inference fails.
   */
  public ListenableFuture<String> generateResponseAsync(
      String inputText, ProgressListener<String> progressListener) {
    LlmInferenceSession session = resetImplicitSession();
    session.addQueryChunk(inputText);
    return session.generateResponseAsync(progressListener);
  }

  /**
   * Runs an invocation of <b>only</b> the tokenization for the LLM, and returns the size (in
   * tokens) of the result. Cannot be called while a {@link #generateResponse(String)} query is
   * active.
   *
   * <p>Note: You cannot invoke simultaneous this operation if any response generations using the
   * same {@link LlmInference} are active. You have to wait for the currently running response
   * generation call to complete before initiating new operations.
   *
   * @param text The text to tokenize.
   * @return The number of tokens in the resulting tokenization of the text.
   * @throws IllegalStateException if the tokenization fails.
   */
  public int sizeInTokens(String text) {
    LlmInferenceSession session = resetImplicitSession();
    return session.sizeInTokens(text);
  }

  public long getSentencePieceProcessorHandle() {
    return taskRunner.getSentencePieceProcessor();
  }

  /** Closes the last implicit session and creates a new one without any existing context. */
  private LlmInferenceSession resetImplicitSession() {
    LlmInferenceSession session = implicitSession.get();
    if (session != null) {
      session.close();
    }

    session =
        LlmInferenceSession.createFromOptions(
            this, LlmInferenceSession.LlmInferenceSessionOptions.builder().build());
    implicitSession.set(session);
    return session;
  }

  LlmTaskRunner getTaskRunner() {
    return taskRunner;
  }

  /** Closes and cleans up the {@link LlmInference}. */
  @Override
  public void close() {
    LlmInferenceSession session = implicitSession.get();
    if (session != null) {
      session.close();
    }
    taskRunner.close();
  }

  /** Options for setting up an {@link LlmInference}. */
  @AutoValue
  public abstract static class LlmInferenceOptions {
    /** Builder for {@link LlmInferenceOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {

      /** Sets the model path for the text generator task. */
      public abstract Builder setModelPath(String modelPath);

      /** Configures the total number of tokens for input and output). */
      public abstract Builder setMaxTokens(int maxTokens);

      /** Configures the maximum number of images to process. Default is 0. */
      public abstract Builder setMaxNumImages(int maxNumImages);

      /**
       * Configures the maximum Top-K value, which is the max Top-K value supported for all sessions
       * created with the engine, used by GPU only. If a session with Top-K value larger than this
       * is being asked to be created, it will be rejected. The default value is 40.
       */
      public abstract Builder setMaxTopK(int maxTopK);

      /** Sets the supported lora ranks for the base model. Used by GPU only. */
      public abstract Builder setSupportedLoraRanks(List<Integer> supportedLoraRanks);

      /** Sets the model options to use for vision modality. */
      public abstract Builder setVisionModelOptions(VisionModelOptions visionModelOptions);

      /** Sets the model options to use for audio modality. */
      public abstract Builder setAudioModelOptions(AudioModelOptions audioModelOptions);

      /** Sets the preferred backend to use for inference. */
      public abstract Builder setPreferredBackend(Backend preferredBackend);

      abstract LlmInferenceOptions autoBuild();

      /** Validates and builds the {@link ImageGeneratorOptions} instance. */
      public final LlmInferenceOptions build() {
        return autoBuild();
      }
    }

    /** The path that points to the tflite model file. */
    public abstract String modelPath();

    /**
     * The total length of the kv-cache. In other words, this is the total number of input + output
     * tokens the model needs to handle.
     */
    public abstract int maxTokens();

    /** The maximum number of images to process. */
    public abstract int maxNumImages();

    /**
     * Returns the maximum Top-K value, which is the max Top-K value supported for all sessions
     * created with the engine, used by GPU only. If a session with Top-K value larger than this is
     * being asked to be created, it will be rejected. The default value is 40.
     */
    public abstract int maxTopK();

    /** The supported lora ranks for the base model. Used by GPU only. */
    public abstract List<Integer> supportedLoraRanks();

    /** The model options to for vision modality. */
    public abstract Optional<VisionModelOptions> visionModelOptions();

    /** The model options to for audio modality. */
    public abstract Optional<AudioModelOptions> audioModelOptions();

    /** Returns the preferred backend to use for inference. */
    public abstract Optional<Backend> preferredBackend();

    /** Returns a new builder with the same values as this instance. */
    public abstract Builder toBuilder();

    /** Instantiates a new LlmInferenceOptions builder. */
    public static Builder builder() {
      return new AutoValue_LlmInference_LlmInferenceOptions.Builder()
          .setMaxTokens(512)
          .setMaxTopK(40)
          .setMaxNumImages(0)
          .setSupportedLoraRanks(Collections.emptyList());
    }
  }
}
