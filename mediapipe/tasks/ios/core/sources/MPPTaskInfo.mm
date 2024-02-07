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
using ::mediapipe::FlowLimiterCalculatorOptions;
using ::mediapipe::InputStreamInfo;
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

- (std::optional<CalculatorGraphConfig>)generateGraphConfigWithError:
    (NSError **)error {
  if ([self.taskOptions respondsToSelector:@selector(copyToProto:)] &&
      [self.taskOptions respondsToSelector:@selector(copyToAnyProto:)]) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInternalError
                          description:@"Only one of copyTo*Proto: methods should be implemented by "
                                      @"the subclass of `MPPTaskOptions`."];
    return std::nullopt;
  }

  CalculatorGraphConfig graphConfig;

  Node *taskSubgraphNode = graphConfig.add_node();
  taskSubgraphNode->set_calculator(self.taskGraphName.cppString);

  if ([self.taskOptions respondsToSelector:@selector(copyToProto:)]) {
    [self.taskOptions copyToProto:taskSubgraphNode->mutable_options()];
  } else if ([self.taskOptions respondsToSelector:@selector(copyToAnyProto:)]) {
    [self.taskOptions copyToAnyProto:taskSubgraphNode->mutable_node_options()->Add()];
  } else {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInternalError
                          description:@"One of copyTo*Proto: methods must be implemented by the "
                                      @"subclass of `MPPTaskOptions`."];
    return std::nullopt;
  }

  for (NSString *outputStream in self.outputStreams) {
    auto cppOutputStream = std::string(outputStream.cppString);
    taskSubgraphNode->add_output_stream(cppOutputStream);
    graphConfig.add_output_stream(cppOutputStream);
  }

  if (!self.enableFlowLimiting) {
    for (NSString *inputStream in self.inputStreams) {
      auto cppInputStream = inputStream.cppString;
      taskSubgraphNode->add_input_stream(cppInputStream);
      graphConfig.add_input_stream(cppInputStream);
    }
    return graphConfig;
  }

  Node *flowLimitCalculatorNode = graphConfig.add_node();

  flowLimitCalculatorNode->set_calculator("FlowLimiterCalculator");

  InputStreamInfo *inputStreamInfo = flowLimitCalculatorNode->add_input_stream_info();
  inputStreamInfo->set_tag_index("FINISHED");
  inputStreamInfo->set_back_edge(true);

  FlowLimiterCalculatorOptions *flowLimitCalculatorOptions =
      flowLimitCalculatorNode->mutable_options()->MutableExtension(
          FlowLimiterCalculatorOptions::ext);
  flowLimitCalculatorOptions->set_max_in_flight(1);
  flowLimitCalculatorOptions->set_max_in_queue(1);

  for (NSString *inputStream in self.inputStreams) {
    graphConfig.add_input_stream(inputStream.cppString);

    NSString *taskInputStream = [MPPTaskInfo addStreamNamePrefix:inputStream];
    taskSubgraphNode->add_input_stream(taskInputStream.cppString);

    NSString *strippedInputStream = [MPPTaskInfo stripTagIndex:inputStream];
    flowLimitCalculatorNode->add_input_stream(strippedInputStream.cppString);

    NSString *strippedTaskInputStream = [MPPTaskInfo stripTagIndex:taskInputStream];
    flowLimitCalculatorNode->add_output_stream(strippedTaskInputStream.cppString);
  }

  NSString *strippedFirstOutputStream = [MPPTaskInfo stripTagIndex:self.outputStreams[0]];
  auto finishedOutputStream = "FINISHED:" + strippedFirstOutputStream.cppString;
  flowLimitCalculatorNode->add_input_stream(finishedOutputStream);

  return graphConfig;
}

+ (NSString *)stripTagIndex:(NSString *)tagIndexName {
  return [tagIndexName componentsSeparatedByString:@":"][1];
}

+ (NSString *)addStreamNamePrefix:(NSString *)tagIndexName {
  NSArray *splits = [tagIndexName componentsSeparatedByString:@":"];
  return [NSString stringWithFormat:@"%@:throttled_%@", splits[0], splits[1]];
}

@end
