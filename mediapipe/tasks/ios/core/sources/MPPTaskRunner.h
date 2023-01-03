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
#include "mediapipe/tasks/cc/core/task_runner.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * This class is used to create and call appropriate methods on the C++ Task Runner.
 */
@interface MPPTaskRunner : NSObject

/**
 * Initializes a new `MPPTaskRunner` with the mediapipe task graph config proto.
 *
 * @param graphConfig A mediapipe task graph config proto.
 *
 * @return An instance of `MPPTaskRunner` initialized to the given graph config proto.
 */
- (instancetype)initWithCalculatorGraphConfig:(mediapipe::CalculatorGraphConfig)graphConfig
                                        error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (absl::StatusOr<mediapipe::tasks::core::PacketMap>)process:
    (const mediapipe::tasks::core::PacketMap &)packetMap;

- (absl::Status)close;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
