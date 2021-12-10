// Copyright 2021 The MediaPipe Authors.
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

package com.google.mediapipe.solutioncore.logging;


import com.google.auto.value.AutoValue;

/** The stats logger interface that defines what MediaPipe solution stats events to log. */
public interface SolutionStatsLogger {
  /** Solution stats snapshot. */
  @AutoValue
  abstract static class StatsSnapshot {
    static StatsSnapshot create(
        int cpuInputCount,
        int gpuInputCount,
        int finishedCount,
        int droppedCount,
        long totalLatencyMs,
        long peakLatencyMs,
        long elapsedTimeMs) {
      return new AutoValue_SolutionStatsLogger_StatsSnapshot(
          cpuInputCount,
          gpuInputCount,
          finishedCount,
          droppedCount,
          totalLatencyMs,
          peakLatencyMs,
          elapsedTimeMs);
    }

    static StatsSnapshot createDefault() {
      return new AutoValue_SolutionStatsLogger_StatsSnapshot(0, 0, 0, 0, 0, 0, 0);
    }

    abstract int cpuInputCount();

    abstract int gpuInputCount();

    abstract int finishedCount();

    abstract int droppedCount();

    abstract long totalLatencyMs();

    abstract long peakLatencyMs();

    abstract long elapsedTimeMs();
  }

  /** Logs the solution session start event. */
  public void logSessionStart();

  /**
   * Records solution API receiving GPU input data.
   *
   * @param packetTimestamp the input packet timestamp that acts as the identifier of the api
   *     invocation.
   */
  public void recordGpuInputArrival(long packetTimestamp);

  /**
   * Records solution API receiving CPU input data.
   *
   * @param packetTimestamp the input packet timestamp that acts as the identifier of the api
   *     invocation.
   */
  public void recordCpuInputArrival(long packetTimestamp);

  /**
   * Records a solution api invocation end event.
   *
   * @param packetTimestamp the output packet timestamp that acts as the identifier of the api
   *     invocation.
   */
  public void recordInvocationEnd(long packetTimestamp);

  /** Logs the solution invocation report event. */
  public void logInvocationReport(StatsSnapshot stats);

  /** Logs the solution session end event. */
  public void logSessionEnd();

  /** Logs the solution init error. */
  public void logInitError();

  /** Logs the solution unsupported input error. */
  public void logUnsupportedInputError();

  /** Logs the solution unsupported output error. */
  public void logUnsupportedOutputError();
}
