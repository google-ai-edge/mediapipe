package com.google.mediapipe.tasks.genai.llminference;

import com.google.auto.value.AutoValue;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.genai.llminference.LlmTaskRunner.LlmSession;
import com.google.mediapipe.tasks.genai.llminference.jni.proto.LlmOptionsProto.LlmSessionConfig;
import com.google.mediapipe.tasks.genai.llminference.jni.proto.LlmResponseContextProto.LlmResponseContext;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * LlmInferenceSession Task Java API.
 *
 * <p>This class encapsulates an LLM inference session along with its context. The context can be
 * shared efficiently across multiple calls to {@link #generateResponse()} or {@link
 * #generateResponseAsync()} and can also be shared across multiple sessions using {@link
 * #cloneSession()}.
 */
public class LlmInferenceSession implements AutoCloseable {
  private static final char TOKEN_SPLITTER = '▁'; // Note this is NOT an underscore: ▁(U+2581)
  private static final String NEW_LINE = "<0x0A>";
  private static final String EOD = "\\[eod\\]";

  private final LlmTaskRunner taskRunner;
  private final LlmSession session;
  private final long callbackHandle;

  private Consumer<LlmResponseContext> currentListener = (unused) -> {};

  /** Constructor to initialize an {@link LlmInferenceSession}. */
  public static LlmInferenceSession createFromOptions(
      LlmInference llmInference, LlmInferenceSessionOptions options) {
    LlmSessionConfig.Builder sessionConfig = LlmSessionConfig.newBuilder();
    sessionConfig.setTopk(options.topK());
    sessionConfig.setTopp(options.topP());
    sessionConfig.setTemperature(options.temperature());
    sessionConfig.setRandomSeed(options.randomSeed());
    if (options.loraPath().isPresent()) {
      sessionConfig.setLoraPath(options.loraPath().get());
    } else {
      sessionConfig.setLoraPath("");
    }

    if (options.graphOptions().isPresent()) {
      GraphOptions graphOptions = options.graphOptions().get();
      LlmSessionConfig.GraphConfig graphConfig =
          LlmSessionConfig.GraphConfig.newBuilder()
              .setIncludeTokenCostCalculator(graphOptions.includeTokenCostCalculator())
              .setEnableVisionModality(graphOptions.enableVisionModality())
              .build();
      sessionConfig.setGraphConfig(graphConfig);
    }

    LlmTaskRunner taskRunner = llmInference.getTaskRunner();
    LlmSession session = taskRunner.createSession(sessionConfig.build());
    return new LlmInferenceSession(taskRunner, session);
  }

  private LlmInferenceSession(LlmTaskRunner taskRunner, LlmSession session) {
    this.taskRunner = taskRunner;
    this.session = session;

    this.callbackHandle =
        taskRunner.registerCallback(
            responseBytes -> currentListener.accept(taskRunner.parseResponse(responseBytes)));
  }

  /**
   * Adds a query chunk to the session. This method can be called multiple times to add multiple
   * query chunks before calling {@link #generateResponse()} or {@link #generateResponseAsync()} .
   * The query chunks will be processed in the order they are added, similar to a concatenated
   * prompt, but able to be processed in chunks.
   *
   * @param inputText The query chunk to add to the session.
   * @throws IllegalStateException if adding a query chunk to the session fails.
   */
  public void addQueryChunk(String inputText) {
    validateState();
    taskRunner.addQueryChunk(session, inputText);
  }

  /**
   * Add an image to the session.
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws IllegalStateException if there is an internal error.
   */
  public void addImage(MPImage image) {
    validateState();
    taskRunner.addImage(session, image);
  }

  /**
   * Generates a response based on the previously added query chunks synchronously. Use {@link
   * #addQueryChunk(String)} to add at least one query chunk before calling this function.
   *
   * <p>Note: You cannot invoke simultaneous response generation calls on active sessions created
   * using the same {@link LlmInference}. You have to wait for the currently running response
   * generation call to complete before initiating another one.
   *
   * @throws IllegalStateException if the inference fails.
   */
  public String generateResponse() {
    validateState();
    List<String> tokens = Collections.unmodifiableList(taskRunner.predictSync(session));
    return decodeResponse(tokens, /* stripLeadingWhitespace= */ true);
  }

  /**
   * Asynchronously generates a response based on the input text. This method cannot be called while
   * other queries are active.
   *
   * <p>The method returns the complete response as a {@link ListenableFuture}. Use {@link
   * #addQueryChunk(String)} to add at least one query chunk before calling this function.
   *
   * <p>Note: You cannot invoke simultaneous response generation calls on active sessions created
   * using the same {@link LlmInference}. You have to wait for the currently running response
   * generation call to complete before initiating another one.
   *
   * @return a {@link ListenableFuture} with the complete response once the inference is complete.
   * @throws IllegalStateException if the inference fails.
   */
  public ListenableFuture<String> generateResponseAsync() {
    validateState();
    return generateResponseAsync((unused1, unused2) -> {});
  }

  /**
   * Asynchronously generates a response based on the input text and emits partial results. This
   * method cannot be called while other queries are active.
   *
   * <p>The method returns the complete response as a {@link ListenableFuture} and invokes the
   * {@code progressListener} as the response is generated. Use {@link #addQueryChunk(String)} to
   * add at least one query chunk before calling this function.
   *
   * <p>Note: You cannot invoke simultaneous response generation calls on active sessions created
   * using the same {@link LlmInference}. You have to wait for the currently running response
   * generation call to complete before initiating another one.
   *
   * @param progressListener a {@link ProgressListener} to receive partial results.
   * @return a {@link ListenableFuture} with the complete response once the inference is complete.
   * @throws IllegalStateException if the inference fails.
   */
  public ListenableFuture<String> generateResponseAsync(ProgressListener<String> progressListener) {
    validateState();

    try {
      taskRunner.acquireLock();

      SettableFuture<String> future = SettableFuture.create();

      currentListener =
          new Consumer<LlmResponseContext>() {
            private final StringBuilder response = new StringBuilder();

            @Override
            public void accept(LlmResponseContext responseContext) {
              boolean done = responseContext.getDone();

              // Not using isEmpty() because it's not available on Android < 30.
              boolean stripLeadingWhitespace = response.length() == 0;
              String partialResultDecoded =
                  decodeResponse(responseContext.getResponsesList(), stripLeadingWhitespace);
              response.append(partialResultDecoded);

              if (done) {
                taskRunner.releaseLock();
                currentListener = unused -> {};
                future.set(response.toString());
              }

              progressListener.run(partialResultDecoded, done);
            }
          };

      taskRunner.predictAsync(session, callbackHandle);
      return future;
    } catch (Throwable t) {
      // Only release the task lock if we fail to start the async inference. For successful
      // inferences, we reset the task lock when we receive `done=true` in the callback above.
      taskRunner.releaseLock();
      this.currentListener = unused -> {};
      throw t;
    }
  }

  /**
   * Runs an invocation of <b>only</b> the tokenization for the LLM, and returns the size (in
   * tokens) of the result. Cannot be called while a other calls are in progress.
   *
   * @param text The text to tokenize.
   * @return The number of tokens in the resulting tokenization of the text.
   * @throws IllegalStateException if the tokenization fails.
   */
  public int sizeInTokens(String text) {
    validateState();
    return taskRunner.sizeInTokens(session, text);
  }

  /** Decodes the response from the LLM engine and returns a human-readable string. */
  private static String decodeResponse(List<String> responses, boolean stripLeadingWhitespace) {
    if (responses.isEmpty()) {
      // Technically, this is an error. We should always get at least one response.
      return "";
    }

    String response = responses.get(0); // We only use the first response
    response = response.replace(TOKEN_SPLITTER, ' '); // Note this is NOT an underscore: ▁(U+2581)
    response = response.replace(NEW_LINE, "\n"); // Replace <0x0A> token with newline

    if (stripLeadingWhitespace) {
      response = stripLeading(response); // Strip all leading spaces for the first output
    }

    return response.split(EOD, -1)[0];
  }

  // TODO: b/368657051 - Make this model public once we can clone GPU sessions on Android.
  /**
   * Clones the current session.
   *
   * <p>You can continue prompting the LLM from where you left off using the cloned session.
   *
   * @return A new instance of `Session` which is cloned from the current session.
   * @throws IllegalStateException if cloning the current session fails.
   */
  LlmInferenceSession cloneSession() {
    LlmSession clonedSession = taskRunner.cloneSession(session);
    return new LlmInferenceSession(taskRunner, clonedSession);
  }

  /** Closes and cleans up the {@link LlmInferenceSession}. */
  @Override
  public void close() {
    validateState();
    taskRunner.unregisterCallback(callbackHandle);
    taskRunner.deleteSession(session);
  }

  private void validateState() {
    if (taskRunner.isLocked()) {
      throw new IllegalStateException("Previous invocation still processing. Wait for done=true.");
    }
  }

  /** Options for setting up an {@link LlmInferenceSessionOptions}. */
  @AutoValue
  public abstract static class LlmInferenceSessionOptions {
    /** Builder for {@link LlmInferenceSessionOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {

      /**
       * Configures the top K number of tokens to be sampled from for each decoding step. A value of
       * 1 means greedy decoding. The default value is 40.
       */
      public abstract Builder setTopK(int topK);

      /**
       * The maximum cumulative probability over the tokens to sample from in each decoding step for
       * top-p / nucleus sampling.
       */
      public abstract Builder setTopP(float topP);

      /**
       * Configures randomness when decoding the next token. A value of 0.0f means greedy decoding.
       * The default value is 0.8f.
       */
      public abstract Builder setTemperature(float temperature);

      /** Configures random seed for sampling tokens. */
      public abstract Builder setRandomSeed(int randomSeed);

      /**
       * The absolute path to the LoRA model asset bundle stored locally on the device. This is only
       * compatible with GPU models.
       */
      public abstract Builder setLoraPath(String loraPath);

      /** Sets the parameters to customize the graph. */
      public abstract Builder setGraphOptions(GraphOptions graphOptions);

      abstract LlmInferenceSessionOptions autoBuild();

      /** Validates and builds the {@link LlmInferenceSessionOptions} instance. */
      public final LlmInferenceSessionOptions build() {
        return autoBuild();
      }
    }

    /**
     * Top K number of tokens to be sampled from for each decoding step. A value of 1 means greedy
     * decoding.
     */
    public abstract int topK();

    /**
     * The maximum cumulative probability over the tokens to sample from in each decoding step for
     * top-p / nucleus sampling.
     */
    public abstract float topP();

    /** Randomness when decoding the next token. A value of 0.0f means greedy decoding. */
    public abstract float temperature();

    /** Random seed for sampling tokens. */
    public abstract int randomSeed();

    /**
     * The absolute path to the LoRA model asset bundle stored locally on the device. This is only
     * compatible with GPU models.
     */
    public abstract Optional<String> loraPath();

    /** Returns the parameters to customize the graph. */
    public abstract Optional<GraphOptions> graphOptions();

    /** Returns a builder with the current options. */
    public abstract Builder toBuilder();

    /** Instantiates a new LlmInferenceOptions builder. */
    public static Builder builder() {
      return new AutoValue_LlmInferenceSession_LlmInferenceSessionOptions.Builder()
          .setTopK(40)
          .setTopP(1.0f)
          .setTemperature(0.8f)
          .setRandomSeed(0);
    }
  }

  static String stripLeading(String text) {
    // stripLeading() implementation for Android < 33
    int left = 0;
    while (left < text.length()) {
      final int codepoint = text.codePointAt(left);
      if (!Character.isWhitespace(codepoint)) {
        break;
      }
      left += Character.charCount(codepoint);
    }
    return text.substring(left);
  }
}
