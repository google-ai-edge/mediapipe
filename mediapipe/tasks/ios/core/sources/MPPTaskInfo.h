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

#import <Foundation/Foundation.h>

#include "mediapipe/framework/calculator.pb.h"

#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskOptionsProtocol.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Holds all needed information to initialize a MediaPipe Task.
 */
@interface MPPTaskInfo : NSObject <NSCopying>

@property(nonatomic, copy, nonnull) NSString *taskGraphName;

/**
 * A task-specific options that is derived from MPPTaskOptions and confirms to
 * MPPTaskOptionsProtocol.
 */
@property(nonatomic, copy) id<MPPTaskOptionsProtocol> taskOptions;

/**
 * List of task graph input stream info strings in the form TAG:name.
 */
@property(nonatomic, copy) NSArray *inputStreams;

/**
 * List of task graph output stream info in the form TAG:name.
 */
@property(nonatomic, copy) NSArray *outputStreams;

/**
 * If the task requires a flow limiter.
 */
@property(nonatomic) BOOL enableFlowLimiting;

+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)initWithTaskGraphName:(NSString *)taskGraphName
                         inputStreams:(NSArray<NSString *> *)inputStreams
                        outputStreams:(NSArray<NSString *> *)outputStreams
                          taskOptions:(id<MPPTaskOptionsProtocol>)taskOptions
                   enableFlowLimiting:(BOOL)enableFlowLimiting
                                error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Creates a MediaPipe Task  protobuf message from the MPPTaskInfo instance.
 */
- (std::optional<::mediapipe::CalculatorGraphConfig>)generateGraphConfigWithError:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
