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
import com.google.mediapipe.tasks.core.jni.proto.LlmOptionsProto.LlmSessionConfig;
import com.google.mediapipe.tasks.core.jni.proto.LlmResponseContextProto.LlmResponseContext;
import com.google.mediapipe.tasks.core.logging.TasksStatsDummyLogger;
import com.google.mediapipe.tasks.core.logging.TasksStatsLogger;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.List;
import java.util.Optional;

/**
 * Internal Task Runner class for all LLM Tasks.
 *
 * @hide
 */
public final class LlmTaskRunner implements AutoCloseable {
  private final long sessionHandle;
  private final Optional<ProgressListener<List<String>>> resultListener;
  private final long callbackHandle;
  private final TasksStatsLogger statsLogger;

  public LlmTaskRunner(
      Context context,
      String taskName,
      LlmSessionConfig sessionConfig,
      Optional<ProgressListener<List<String>>> resultListener) {
    statsLogger = TasksStatsDummyLogger.create(context, taskName, /* taskRunningModeStr= */ "");
    this.sessionHandle = nativeCreateSession(sessionConfig.toByteArray());

    this.resultListener = resultListener;
    if (resultListener.isPresent()) {
      this.callbackHandle = nativeRegisterCallback(this);
    } else {
      this.callbackHandle = 0;
    }
    statsLogger.logSessionStart();
  }

  /** Invokes the LLM with the provided input and waits for the result. */
  public List<String> predictSync(String input) {
    byte[] responseBytes = nativePredictSync(sessionHandle, input);
    return parseResponse(responseBytes).getResponsesList();
  }

  /** Invokes the LLM with the provided input and calls the callback with the result. */
  public void predictAsync(String input) {
    if (callbackHandle == 0) {
      throw new IllegalStateException("No result listener provided.");
    }
    nativePredictAsync(sessionHandle, callbackHandle, input);
  }

  /** Invokes the native token cost calculator and returns the size of the string in tokens. */
  public int sizeInTokens(String text) {
    return nativeSizeInTokens(sessionHandle, text);
  }

  private LlmResponseContext parseResponse(byte[] response) {
    try {
      return LlmResponseContext.parseFrom(response);
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalStateException("Failed to parse response", e);
    }
  }

  private void onAsyncResponse(byte[] responseBytes) {
    LlmResponseContext respone = parseResponse(responseBytes);
    resultListener.get().run(respone.getResponsesList(), respone.getDone());
  }

  @Override
  public void close() {
    if (callbackHandle != 0) {
      nativeRemoveCallback(callbackHandle);
    }
    nativeDeleteSession(sessionHandle);
    statsLogger.logSessionEnd();
  }

  private static native long nativeCreateSession(byte[] sessionConfig);

  private static native void nativeDeleteSession(long sessionPointer);

  private static native byte[] nativePredictSync(long sessionPointer, String input);

  private static native long nativeRegisterCallback(Object callback);

  private static native void nativeRemoveCallback(long callbackHandle);

  private static native void nativePredictAsync(
      long sessionPointer, long callbackContextHandle, String input);

  private static native int nativeSizeInTokens(long sessionPointer, String input);
}
