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

package com.google.mediapipe.tasks.core;

import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig;
import com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.Node;
import com.google.mediapipe.proto.CalculatorProto.InputStreamInfo;
import com.google.mediapipe.calculator.proto.FlowLimiterCalculatorProto.FlowLimiterCalculatorOptions;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.protobuf.Any;
import java.util.ArrayList;
import java.util.List;

/**
 * {@link TaskInfo} contains all needed information to initialize a MediaPipe Task {@link
 * com.google.mediapipe.framework.Graph}.
 */
@AutoValue
public abstract class TaskInfo<T extends TaskOptions> {
  /** Builder for {@link TaskInfo}. */
  @AutoValue.Builder
  public abstract static class Builder<T extends TaskOptions> {
    /** Sets the MediaPipe task name. */
    public abstract Builder<T> setTaskName(String value);

    /** Sets the MediaPipe task running mode name. */
    public abstract Builder<T> setTaskRunningModeName(String value);

    /** Sets the MediaPipe task graph name. */
    public abstract Builder<T> setTaskGraphName(String value);

    /** Sets a list of task graph input stream info {@link String}s in the form TAG:name. */
    public abstract Builder<T> setInputStreams(List<String> value);

    /** Sets a list of task graph output stream info {@link String}s in the form TAG:name. */
    public abstract Builder<T> setOutputStreams(List<String> value);

    /** Sets to true if the task requires a flow limiter. */
    public abstract Builder<T> setEnableFlowLimiting(boolean value);

    /**
     * Sets a task-specific options instance.
     *
     * @param value a task-specific options that is derived from {@link TaskOptions}.
     */
    public abstract Builder<T> setTaskOptions(T value);

    public abstract TaskInfo<T> autoBuild();

    /**
     * Validates and builds the {@link TaskInfo} instance. *
     *
     * @throws IllegalArgumentException if the required information such as task graph name, graph
     *     input streams, and the graph output streams are empty.
     */
    public final TaskInfo<T> build() {
      TaskInfo<T> taskInfo = autoBuild();
      if (taskInfo.taskGraphName().isEmpty()
          || taskInfo.inputStreams().isEmpty()
          || taskInfo.outputStreams().isEmpty()) {
        throw new IllegalArgumentException(
            "Task graph's name, input streams, and output streams should be non-empty.");
      }
      return taskInfo;
    }
  }

  abstract String taskName();

  abstract String taskRunningModeName();

  abstract String taskGraphName();

  abstract T taskOptions();

  abstract List<String> inputStreams();

  abstract List<String> outputStreams();

  abstract boolean enableFlowLimiting();

  public static <T extends TaskOptions> Builder<T> builder() {
    return new AutoValue_TaskInfo.Builder<T>().setTaskName("").setTaskRunningModeName("");
  }

  /* Returns a list of the output stream names without the stream tags. */
  List<String> outputStreamNames() {
    List<String> streamNames = new ArrayList<>(outputStreams().size());
    for (String stream : outputStreams()) {
      streamNames.add(stream.substring(stream.lastIndexOf(':') + 1));
    }
    return streamNames;
  }

  /**
   * Creates a MediaPipe Task {@link CalculatorGraphConfig} protobuf message from the {@link
   * TaskInfo} instance.
   */
  CalculatorGraphConfig generateGraphConfig() {
    CalculatorGraphConfig.Builder graphBuilder = CalculatorGraphConfig.newBuilder();
    CalculatorOptions options = taskOptions().convertToCalculatorOptionsProto();
    Any anyOptions = taskOptions().convertToAnyProto();
    if (!(options == null ^ anyOptions == null)) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INVALID_ARGUMENT.ordinal(),
          "Only one of convertTo*Proto() method should be implemented for "
              + taskOptions().getClass());
    }
    Node.Builder taskSubgraphBuilder = Node.newBuilder().setCalculator(taskGraphName());
    if (options != null) {
      taskSubgraphBuilder.setOptions(options);
    }
    if (anyOptions != null) {
      taskSubgraphBuilder.addNodeOptions(anyOptions);
    }
    for (String outputStream : outputStreams()) {
      taskSubgraphBuilder.addOutputStream(outputStream);
      graphBuilder.addOutputStream(outputStream);
    }
    if (!enableFlowLimiting()) {
      for (String inputStream : inputStreams()) {
        taskSubgraphBuilder.addInputStream(inputStream);
        graphBuilder.addInputStream(inputStream);
      }
      graphBuilder.addNode(taskSubgraphBuilder.build());
      return graphBuilder.build();
    }
    Node.Builder flowLimiterCalculatorBuilder =
        Node.newBuilder()
            .setCalculator("FlowLimiterCalculator")
            .addInputStreamInfo(
                InputStreamInfo.newBuilder().setTagIndex("FINISHED").setBackEdge(true).build())
            .setOptions(
                CalculatorOptions.newBuilder()
                    .setExtension(
                        FlowLimiterCalculatorOptions.ext,
                        FlowLimiterCalculatorOptions.newBuilder()
                            .setMaxInFlight(1)
                            .setMaxInQueue(1)
                            .build())
                    .build());
    for (String inputStream : inputStreams()) {
      graphBuilder.addInputStream(inputStream);
      flowLimiterCalculatorBuilder.addInputStream(stripTagIndex(inputStream));
      String taskInputStream = addStreamNamePrefix(inputStream);
      flowLimiterCalculatorBuilder.addOutputStream(stripTagIndex(taskInputStream));
      taskSubgraphBuilder.addInputStream(taskInputStream);
    }
    flowLimiterCalculatorBuilder.addInputStream(
        "FINISHED:" + stripTagIndex(outputStreams().get(0)));
    graphBuilder.addNode(flowLimiterCalculatorBuilder.build());
    graphBuilder.addNode(taskSubgraphBuilder.build());
    return graphBuilder.build();
  }

  private String stripTagIndex(String tagIndexName) {
    return tagIndexName.substring(tagIndexName.lastIndexOf(':') + 1);
  }

  private String addStreamNamePrefix(String tagIndexName) {
    return tagIndexName.substring(0, tagIndexName.lastIndexOf(':') + 1)
        + "throttled_"
        + stripTagIndex(tagIndexName);
  }
}
