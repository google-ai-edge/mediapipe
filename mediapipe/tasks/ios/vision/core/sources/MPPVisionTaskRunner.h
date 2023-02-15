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
#import "mediapipe/tasks/ios/vision/core/sources/MPPRunningMode.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * This class is used to create and call appropriate methods on the C++ Task Runner to initialize,
 * execute and terminate any MediaPipe vision task.
 */
@interface MPPVisionTaskRunner : MPPTaskRunner

/**
 * Initializes a new `MPPVisionTaskRunner` with the MediaPipe calculator config proto running mode
 * and packetsCallback.
 * Make sure that the packets callback is set properly based on the vision task's running mode.
 * In case of live stream running mode, a C++ packets callback that is intended to deliver inference
 * results must be provided. In case of image or video running mode, packets callback must be set to
 * nil.
 *
 * @param graphConfig A MediaPipe calculator config proto.
 * @param runningMode MediaPipe vision task running mode.
 * @param packetsCallback An optional C++ callback function that takes a list of output packets as
 * the input argument. If provided, the callback must in turn call the block provided by the user in
 * the appropriate task options. Make sure that the packets callback is set properly based on the
 * vision task's running mode. In case of live stream running mode, a C++ packets callback that is
 * intended to deliver inference results must be provided. In case of image or video running mode,
 * packets callback must be set to nil.
 *
 * @param error Pointer to the memory location where errors if any should be
 * saved. If @c NULL, no error will be saved.
 *
 * @return An instance of `MPPVisionTaskRunner` initialized to the given MediaPipe calculator config
 * proto, running mode and packets callback.
 */
- (nullable instancetype)initWithCalculatorGraphConfig:(mediapipe::CalculatorGraphConfig)graphConfig
                                           runningMode:(MPPRunningMode)runningMode
                                       packetsCallback:
                                           (mediapipe::tasks::core::PacketsCallback)packetsCallback
                                                 error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
