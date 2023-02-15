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

#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

namespace {
using ::mediapipe::CalculatorGraphConfig;
}  // namespace

@interface MPPVisionTaskRunner () {
  MPPRunningMode _runningMode;
}
@end

@implementation MPPVisionTaskRunner

- (nullable instancetype)initWithCalculatorGraphConfig:(mediapipe::CalculatorGraphConfig)graphConfig
                                           runningMode:(MPPRunningMode)runningMode
                                       packetsCallback:
                                           (mediapipe::tasks::core::PacketsCallback)packetsCallback
                                                 error:(NSError **)error {
  switch (runningMode) {
    case MPPRunningModeImage:
    case MPPRunningModeVideo: {
      if (packetsCallback) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInvalidArgumentError
                              description:@"The vision task is in image or video mode, a "
                                          @"user-defined result callback should not be provided."];
        return nil;
      }
      break;
    }
    case MPPRunningModeLiveStream: {
      if (!packetsCallback) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInvalidArgumentError
                              description:@"The vision task is in live stream mode, a user-defined "
                                          @"result callback must be provided."];
        return nil;
      }
      break;
    }
    default: {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Unrecognized running mode"];
      return nil;
    }
  }

  _runningMode = runningMode;
  self = [super initWithCalculatorGraphConfig:graphConfig
                              packetsCallback:packetsCallback
                                        error:error];
  return self;
}

@end
