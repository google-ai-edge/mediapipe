/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"

#include "mediapipe/calculators/core/flow_limiter_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_options.pb.h"

namespace {
using CalculatorGraphConfig = ::mediapipe::CalculatorGraphConfig;
using Node = ::mediapipe::CalculatorGraphConfig::Node;
using InputStreamInfo = ::mediapipe::InputStreamInfo;
using CalculatorOptions = ::mediapipe::CalculatorOptions;
using FlowLimiterCalculatorOptions = ::mediapipe::FlowLimiterCalculatorOptions;
}  // namespace

@implementation MPPTaskInfo

- (instancetype)initWithTaskGraphName:(NSString *)taskGraphName
                         inputStreams:(NSArray<NSString *> *)inputStreams
                        outputStreams:(NSArray<NSString *> *)outputStreams
                          taskOptions:(id<MPPTaskOptionsProtocol>)taskOptions
                   enableFlowLimiting:(BOOL)enableFlowLimiting
                                error:(NSError **)error {
  if (!taskGraphName || !inputStreams.count || !outputStreams.count) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:
                  @"Task graph's name, input streams, and output streams should be non-empty."];
  }

  self = [super init];

  if (self) {
    _taskGraphName = taskGraphName;
    _inputStreams = inputStreams;
    _outputStreams = outputStreams;
    _taskOptions = taskOptions;
    _enableFlowLimiting = enableFlowLimiting;
  }
  return self;
}

- (id)copyWithZone:(NSZone *)zone {
  MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc] init];

  taskInfo.taskGraphName = self.taskGraphName;
  taskInfo.inputStreams = self.inputStreams;
  taskInfo.outputStreams = self.outputStreams;
  taskInfo.taskOptions = self.taskOptions;
  taskInfo.enableFlowLimiting = self.enableFlowLimiting;

  return taskInfo;
}

- (CalculatorGraphConfig)generateGraphConfig {
  CalculatorGraphConfig graph_config;

  Node *task_subgraph_node = graph_config.add_node();
  task_subgraph_node->set_calculator(self.taskGraphName.cppString);
  [self.taskOptions copyToProto:task_subgraph_node->mutable_options()];

  for (NSString *outputStream in self.outputStreams) {
    auto cpp_output_stream = std::string(outputStream.cppString);
    task_subgraph_node->add_output_stream(cpp_output_stream);
    graph_config.add_output_stream(cpp_output_stream);
  }

  if (self.enableFlowLimiting) {
    Node *flow_limit_calculator_node = graph_config.add_node();

    flow_limit_calculator_node->set_calculator("FlowLimiterCalculator");

    InputStreamInfo *input_stream_info = flow_limit_calculator_node->add_input_stream_info();
    input_stream_info->set_tag_index("FINISHED");
    input_stream_info->set_back_edge(true);

    FlowLimiterCalculatorOptions *flow_limit_calculator_options =
        flow_limit_calculator_node->mutable_options()->MutableExtension(
            FlowLimiterCalculatorOptions::ext);
    flow_limit_calculator_options->set_max_in_flight(1);
    flow_limit_calculator_options->set_max_in_queue(1);

    for (NSString *inputStream in self.inputStreams) {
      graph_config.add_input_stream(inputStream.cppString);

      NSString *strippedInputStream = [MPPTaskInfo stripTagIndex:inputStream];
      flow_limit_calculator_node->add_input_stream(strippedInputStream.cppString);

      NSString *taskInputStream = [MPPTaskInfo addStreamNamePrefix:inputStream];
      task_subgraph_node->add_input_stream(taskInputStream.cppString);

      NSString *strippedTaskInputStream = [MPPTaskInfo stripTagIndex:taskInputStream];
      flow_limit_calculator_node->add_output_stream(strippedTaskInputStream.cppString);
    }

    NSString *firstOutputStream = self.outputStreams[0];
    auto finished_output_stream = "FINISHED:" + firstOutputStream.cppString;
    flow_limit_calculator_node->add_input_stream(finished_output_stream);
  } else {
    for (NSString *inputStream in self.inputStreams) {
      auto cpp_input_stream = inputStream.cppString;
      task_subgraph_node->add_input_stream(cpp_input_stream);
      graph_config.add_input_stream(cpp_input_stream);
    }
  }

  return graph_config;
}

+ (NSString *)stripTagIndex:(NSString *)tagIndexName {
  return [tagIndexName componentsSeparatedByString:@":"][1];
}

+ (NSString *)addStreamNamePrefix:(NSString *)tagIndexName {
  NSArray *splits = [tagIndexName componentsSeparatedByString:@":"];
  return [NSString stringWithFormat:@"%@:throttled_%@", splits[0], splits[1]];
}

@end
