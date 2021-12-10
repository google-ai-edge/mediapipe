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

import android.content.Context;
import com.google.mediapipe.solutioncore.SolutionInfo;

/** A dummy solution stats logger that logs nothing. */
public class SolutionStatsDummyLogger implements SolutionStatsLogger {

  /**
   * Initializes the solution stats dummy logger.
   *
   * @param context a {@link Context}.
   * @param solutionNameStr the solution name.
   * @param solutionInfo a {@link SolutionInfo}.
   */
  public SolutionStatsDummyLogger(
      Context context, String solutionNameStr, SolutionInfo solutionInfo) {}

  /** Logs the solution session start event. */
  @Override
  public void logSessionStart() {}

  /**
   * Records solution API receiving GPU input data.
   *
   * @param packetTimestamp the input packet timestamp that acts as the identifier of the api
   *     invocation.
   */
  @Override
  public void recordGpuInputArrival(long packetTimestamp) {}

  /**
   * Records solution API receiving CPU input data.
   *
   * @param packetTimestamp the input packet timestamp that acts as the identifier of the api
   *     invocation.
   */
  @Override
  public void recordCpuInputArrival(long packetTimestamp) {}

  /**
   * Records a solution api invocation end event.
   *
   * @param packetTimestamp the output packet timestamp that acts as the identifier of the api
   *     invocation.
   */
  @Override
  public void recordInvocationEnd(long packetTimestamp) {}

  /** Logs the solution invocation report event. */
  @Override
  public void logInvocationReport(StatsSnapshot stats) {}

  /** Logs the solution session end event. */
  @Override
  public void logSessionEnd() {}

  /** Logs the solution init error. */
  @Override
  public void logInitError() {}

  /** Logs the solution unsupported input error. */
  @Override
  public void logUnsupportedInputError() {}

  /** Logs the solution unsupported output error. */
  @Override
  public void logUnsupportedOutputError() {}
}
