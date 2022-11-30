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
#import "mediapipe/tasks/ios/text/core/sources/MPPBaseTextTaskApi.h"

#include "mediapipe/tasks/cc/core/task_runner.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

namespace {
using ::mediapipe::CalculatorGraphConfig;
using TaskRunnerCpp = ::mediapipe::tasks::core::TaskRunner;
}  // namespace

@interface MPPBaseTextTaskApi () {
  /** TextSearcher backed by C++ API */
  std::unique_ptr<TaskRunnerCpp> _taskRunner;
}
@end

@implementation MPPBaseTextTaskApi

- (instancetype)initWithCalculatorGraphConfig:(CalculatorGraphConfig)graphConfig
                                        error:(NSError **)error {
  self = [super init];
  if (self) {
    auto taskRunnerResult = TaskRunnerCpp::Create(std::move(graphConfig));

    if (![MPPCommonUtils checkCppError:taskRunnerResult.status() toError:error]) {
      return nil;
    }

    _taskRunner = std::move(taskRunnerResult.value());
  }
  return self;
}

- (void)close {
  _taskRunner->Close();
}

@end
