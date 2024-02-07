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

#import "mediapipe/tasks/ios/text/core/sources/MPPTextTaskRunner.h"

namespace {
using ::mediapipe::CalculatorGraphConfig;
}  // namespace

@implementation MPPTextTaskRunner

- (instancetype)initWithTaskInfo:(MPPTaskInfo *)taskInfo
                 packetsCallback:(mediapipe::tasks::core::PacketsCallback)packetsCallback
                           error:(NSError **)error {
  self = [super initWithTaskInfo:taskInfo packetsCallback:packetsCallback error:error];
  return self;
}

- (instancetype)initWithTaskInfo:(MPPTaskInfo *)taskInfo error:(NSError **)error {
  return [self initWithTaskInfo:taskInfo packetsCallback:nullptr error:error];
}

@end
