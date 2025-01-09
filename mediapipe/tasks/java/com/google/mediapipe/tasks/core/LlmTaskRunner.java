// Copyright 2023 The MediaPipe Authors.
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

import android.content.Context;
import com.google.mediapipe.tasks.core.OutputHandler.ProgressListener;
import com.google.mediapipe.tasks.core.jni.proto.LlmOptionsProto.LlmModelSettings;
import com.google.mediapipe.tasks.core.jni.proto.LlmOptionsProto.LlmSessionConfig;
import com.google.mediapipe.tasks.core.jni.proto.LlmResponseContextProto.LlmResponseContext;
import com.google.mediapipe.tasks.core.logging.TasksStatsDummyLogger;
import com.google.mediapipe.tasks.core.logging.TasksStatsLogger;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Internal Task Runner class for all LLM Tasks.
 *
 * @hide
 */
public final class LlmTaskRunner implements AutoCloseable {
  private final long engineHandle;
  private final Optional<ProgressListener<List<String>>> resultListener;
  private final long callbackHandle;
  private final TasksStatsLogger statsLogger;
  private final AtomicBoolean isProcessing;

  /** The session to use for LLM inference calls. */
  public static final class LlmSession {
    private final long sessionHandle;

    LlmSession(long sessionHandle) {
      this.sessionHandle = sessionHandle;
    }
  }

  public LlmTaskRunner(
      Context context,
      String taskName,
      LlmModelSettings modelSettings,
      Optional<ProgressListener<List<String>>> resultListener) {
    statsLogger = TasksStatsDummyLogger.create(context, taskName, /* taskRunningModeStr= */ "");
    this.engineHandle = nativeCreateEngine(modelSettings.toByteArray());

    this.resultListener = resultListener;
    if (resultListener.isPresent()) {
      this.callbackHandle = nativeRegisterCallback(this);
    } else {
      this.callbackHandle = 0;
    }

    this.isProcessing = new AtomicBoolean(false);
  }

  /** Creates a new LLM session. */
  public LlmSession createSession(LlmSessionConfig sessionConfig) {
    validateState();
    long sessionHandle = nativeCreateSession(sessionConfig.toByteArray(), engineHandle);
    statsLogger.logSessionStart();
    return new LlmSession(sessionHandle);
  }

  /** Adds a new query to the session context. */
  public void addQueryChunk(LlmSession session, String input) {
    validateState();
    nativeAddQueryChunk(session.sessionHandle, input);
  }

  /** Invokes the LLM with the given session and waits for the result. */
  public List<String> predictSync(LlmSession session) {
    validateState();
    try {
      isProcessing.set(true);
      byte[] responseBytes = nativePredictSync(session.sessionHandle);
      return parseResponse(responseBytes).getResponsesList();
    } finally {
      isProcessing.set(false);
    }
  }

  /** Invokes the LLM with the given session and calls the callback with the result. */
  public void predictAsync(LlmSession session) {
    validateState();

    if (callbackHandle == 0) {
      throw new IllegalStateException("No result listener provided.");
    }

    try {
      isProcessing.set(true);
      nativePredictAsync(session.sessionHandle, callbackHandle);
    } catch (Throwable t) {
      // Only reset `isProcessing` if we fail to start the async inference. For successful
      // inferences, we reset `isProcessing` when we receive `done=true`.
      isProcessing.set(false);
      throw t;
    }
  }

  /** Invokes the native token cost calculator and returns the size of the string in tokens. */
  public int sizeInTokens(LlmSession session, String text) {
    validateState();
    try {
      isProcessing.set(true);
      return nativeSizeInTokens(session.sessionHandle, text);
    } finally {
      isProcessing.set(false);
    }
  }

  /** Clones the current session. */
  public LlmSession cloneSession(LlmSession session) {
    validateState();
    long clonedSessionHandle = nativeCloneSession(session.sessionHandle);
    statsLogger.logSessionClone();
    return new LlmSession(clonedSessionHandle);
  }

  /** Removes the session and frees up its context. */
  public void deleteSession(LlmSession session) {
    validateState();
    nativeDeleteSession(session.sessionHandle);
    statsLogger.logSessionEnd();
  }

  private LlmResponseContext parseResponse(byte[] response) {
    try {
      return LlmResponseContext.parseFrom(response);
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalStateException("Failed to parse response", e);
    }
  }

  private void onAsyncResponse(byte[] responseBytes) {
    LlmResponseContext response = parseResponse(responseBytes);
    if (response.getDone()) {
      isProcessing.set(false);
    }
    resultListener.get().run(response.getResponsesList(), response.getDone());
  }

  @Override
  public void close() {
    validateState();
    if (callbackHandle != 0) {
      nativeRemoveCallback(callbackHandle);
    }
  }

  private void validateState() {
    if (isProcessing.get()) {
      throw new IllegalStateException("Previous invocation still processing. Wait for done=true.");
    }
  }

  private static native long nativeCreateEngine(byte[] modelSettings);

  private static native void nativeDeleteEngine(long enginePointer);

  private static native long nativeCreateSession(byte[] sessionConfig, long enginePointer);

  private static native long nativeCloneSession(long sessionPointer);

  private static native void nativeDeleteSession(long sessionPointer);

  private static native void nativeAddQueryChunk(long sessionPointer, String input);

  private static native byte[] nativePredictSync(long sessionPointer);

  private static native long nativeRegisterCallback(Object callback);

  private static native void nativeRemoveCallback(long callbackHandle);

  private static native void nativePredictAsync(long sessionPointer, long callbackContextHandle);

  private static native int nativeSizeInTokens(long sessionPointer, String input);
}
