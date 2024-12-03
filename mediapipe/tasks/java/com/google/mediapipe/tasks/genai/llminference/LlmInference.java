package com.google.mediapipe.tasks.genai.llminference;

import static com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession.decodeResponse;

import android.content.Context;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.tasks.core.ErrorListener;
import com.google.mediapipe.tasks.core.LlmTaskRunner;
import com.google.mediapipe.tasks.core.OutputHandler.ProgressListener;
import com.google.mediapipe.tasks.core.TaskOptions;
import com.google.mediapipe.tasks.core.jni.proto.LlmOptionsProto.LlmModelSettings;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

/** LlmInference Task Java API */
public final class LlmInference implements AutoCloseable {
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
    LlmModelSettings modelSettings =
        LlmModelSettings.newBuilder()
            .setModelPath(options.modelPath())
            .setCacheDir(context.getCacheDir().getAbsolutePath())
            .setNumDecodeStepsPerSync(NUM_DECODE_STEPS_PER_SYNC)
            .setMaxTokens(options.maxTokens())
            .setMaxTopK(options.maxTopK())
            .setNumberOfSupportedLoraRanks(options.supportedLoraRanks().size())
            .addAllSupportedLoraRanks(options.supportedLoraRanks())
            .build();

    return new LlmInference(context, STATS_TAG, modelSettings, options.resultListener());
  }

  /** Constructor to initialize an {@link LlmInference}. */
  private LlmInference(
      Context context,
      String taskName,
      LlmModelSettings modelSettings,
      Optional<ProgressListener<String>> resultListener) {
    Optional<ProgressListener<List<String>>> llmResultListener;
    if (resultListener.isPresent()) {
      llmResultListener =
          Optional.of(
              new ProgressListener<List<String>>() {
                private boolean receivedFirstToken = false;

                @Override
                public void run(List<String> partialResult, boolean done) {
                  String result =
                      decodeResponse(
                          partialResult, /* stripLeadingWhitespace= */ !receivedFirstToken);
                  if (done) {
                    receivedFirstToken = false; // Reset to initial state
                    resultListener.get().run(result, done);
                  } else if (!result.isEmpty()) {
                    receivedFirstToken = true;
                    resultListener.get().run(result, done);
                  }
                }
              });
    } else {
      llmResultListener = Optional.empty();
    }

    this.taskRunner = new LlmTaskRunner(context, taskName, modelSettings, llmResultListener);
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
   * @throws MediaPipeException if the inference fails.
   */
  public String generateResponse(String inputText) {
    LlmInferenceSession session = resetImplicitSession();
    session.addQueryChunk(inputText);
    return session.generateResponse();
  }

  /**
   * Generates a response based on the input text. This method cannot be called while other queries
   * are active.
   *
   * <p>This function creates a new session for each call. If you want to have a stateful inference,
   * use {@link LlmInferenceSession#generateResponseAsync()} instead.
   *
   * <p>Note: You cannot invoke simultaneous response generation calls on active sessions created
   * using the same {@link LlmInference}. You have to wait for the currently running response
   * generation call to complete before initiating another one.
   *
   * @param inputText a {@link String} for processing.
   * @throws MediaPipeException if the inference fails.
   */
  public void generateResponseAsync(String inputText) {
    LlmInferenceSession session = resetImplicitSession();
    session.addQueryChunk(inputText);
    session.generateResponseAsync();
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
   * @throws MediaPipeException if the tokenization fails.
   */
  public int sizeInTokens(String text) {
    LlmInferenceSession session = resetImplicitSession();
    return session.sizeInTokens(text);
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
  public abstract static class LlmInferenceOptions extends TaskOptions {
    /** Builder for {@link LlmInferenceOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {

      /** Sets the model path for the text generator task. */
      public abstract Builder setModelPath(String modelPath);

      /** Sets the result listener to invoke with the async API. */
      public abstract Builder setResultListener(ProgressListener<String> listener);

      /** Sets the error listener to invoke with the async API. */
      public abstract Builder setErrorListener(ErrorListener listener);

      /** Configures the total number of tokens for input and output). */
      public abstract Builder setMaxTokens(int maxTokens);

      /**
       * Configures the maximum Top-K value, which is the max Top-K value supported for all sessions
       * created with the engine, used by GPU only. If a session with Top-K value larger than this
       * is being asked to be created, it will be rejected. The default value is 40.
       */
      public abstract Builder setMaxTopK(int maxTopK);

      /** The supported lora ranks for the base model. Used by GPU only. */
      public abstract Builder setSupportedLoraRanks(List<Integer> supportedLoraRanks);

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

    /**
     * Returns the maximum Top-K value, which is the max Top-K value supported for all sessions
     * created with the engine, used by GPU only. If a session with Top-K value larger than this is
     * being asked to be created, it will be rejected. The default value is 40.
     */
    public abstract int maxTopK();

    /** The supported lora ranks for the base model. Used by GPU only. */
    public abstract List<Integer> supportedLoraRanks();

    /** The result listener to use for the {@link LlmInference#generateAsync} API. */
    public abstract Optional<ProgressListener<String>> resultListener();

    /** The error listener to use for the {@link LlmInference#generateAsync} API. */
    public abstract Optional<ErrorListener> errorListener();

    /** Instantiates a new LlmInferenceOptions builder. */
    public static Builder builder() {
      return new AutoValue_LlmInference_LlmInferenceOptions.Builder()
          .setMaxTokens(512)
          .setMaxTopK(40)
          .setSupportedLoraRanks(Collections.emptyList());
    }
  }
}
