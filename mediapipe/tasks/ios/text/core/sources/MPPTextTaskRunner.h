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

#import <Foundation/Foundation.h>
#import "mediapipe/tasks/ios/core/sources/MPPTaskRunner.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * This class is used to create and call appropriate methods on the C++ Task Runner to initialize,
 * execute and terminate any MediaPipe text task.
 */
@interface MPPTextTaskRunner : MPPTaskRunner

/**
 * Initializes a new `MPPTextTaskRunner` with the MediaPipe calculator config proto.
 *
 * @param taskInfo A `MPPTaskInfo` initialized by the task.
 *
 * @return An instance of `MPPTextTaskRunner` initialized with the given task info.
 */
- (instancetype)initWithTaskInfo:(MPPTaskInfo *)taskInfo error:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
