// Copyright 2022 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.core.logging;

import android.content.Context;

/** A dummy MediaPipe Tasks stats logger that has all methods as no-ops. */
public class TasksStatsDummyLogger implements TasksStatsLogger {

  /**
   * Creates the MediaPipe Tasks stats dummy logger.
   *
   * @param context a {@link Context}.
   * @param taskNameStr the task api name.
   * @param taskRunningModeStr the task running mode string representation.
   */
  public static TasksStatsDummyLogger create(
      Context context, String taskNameStr, String taskRunningModeStr) {
    return new TasksStatsDummyLogger();
  }

  private TasksStatsDummyLogger() {}

  /** Logs the start of a MediaPipe Tasks API session. */
  @Override
  public void logSessionStart() {}

  /** Logs the cloning of a MediaPipe Tasks API session. */
  @Override
  public void logSessionClone() {}

  /**
   * Records MediaPipe Tasks API receiving CPU input data.
   *
   * @param packetTimestamp the input packet timestamp that acts as the identifier of the api
   *     invocation.
   */
  @Override
  public void recordCpuInputArrival(long packetTimestamp) {}

  /**
   * Records MediaPipe Tasks API receiving GPU input data.
   *
   * @param packetTimestamp the input packet timestamp that acts as the identifier of the api
   *     invocation.
   */
  @Override
  public void recordGpuInputArrival(long packetTimestamp) {}

  /**
   * Records the end of a Mediapipe Tasks API invocation.
   *
   * @param packetTimestamp the output packet timestamp that acts as the identifier of the api
   *     invocation.
   */
  @Override
  public void recordInvocationEnd(long packetTimestamp) {}

  /** Logs the MediaPipe Tasks API periodic invocation report. */
  @Override
  public void logInvocationReport(StatsSnapshot stats) {}

  /** Logs the Tasks API session end event. */
  @Override
  public void logSessionEnd() {}

  /** Logs the MediaPipe Tasks API initialization error. */
  @Override
  public void logInitError() {}
}
