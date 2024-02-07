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

#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"

#include <optional>

NS_ASSUME_NONNULL_BEGIN

/**
 * This class is used to create and call appropriate methods on the C++ Task Runner to initialize,
 * execute and terminate any MediaPipe task.
 *
 * An instance of the newly created C++ task runner will be stored until this class is destroyed.
 * When methods are called for processing (performing inference), closing etc., on this class,
 * internally the appropriate methods will be called on the C++ task runner instance to execute the
 * appropriate actions. For each type of task, a subclass of this class must be defined to add any
 * additional functionality. For eg:, vision tasks must create an `MPPVisionTaskRunner` and provide
 * additional functionality. An instance of `MPPVisionTaskRunner` can in turn be used by the each
 * vision task for creation and execution of the task. Please see the documentation for the C++ Task
 * Runner for more details on how the tasks runner operates.
 */
@interface MPPTaskRunner : NSObject

/**
 * The canonicalized `CalculatorGraphConfig` of the underlying graph managed by the C++ task
 * runner.
 */
@property(nonatomic, readonly) const mediapipe::CalculatorGraphConfig &graphConfig;

/**
 * Initializes a new `MPPTaskRunner` with the given task info and an optional C++ packets callback.
 *
 * You can pass `nullptr` for `packetsCallback` in case the mode of operation requested by the user
 * is synchronous.
 *
 * If the task is operating in asynchronous mode, any iOS MediaPipe task that uses the
 * `MPPTaskRunner` must define a C++ callback function to obtain the results of inference
 * asynchronously and deliver the results to the user. To accomplish this, the callback function
 * should in turn invoke the block provided by the user in the task options supplied to create the
 * task. Please see the documentation of the C++ Task Runner for more information on the synchronous
 * and asynchronous modes of operation.
 *
 * @param taskInfo A `MPPTaskInfo` initialized by the task.
 * @param packetsCallback An optional C++ callback function that takes a list of output packets as
 * the input argument. If provided, the callback must in turn call the block provided by the user in
 * the appropriate task options.
 *
 * @return An instance of `MPPTaskRunner` initialized with the given task info and the optional C++
 * packets callback.
 */
- (instancetype)initWithTaskInfo:(MPPTaskInfo *)taskInfo
                 packetsCallback:(mediapipe::tasks::core::PacketsCallback)packetsCallback
                           error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * A synchronous method for invoking the C++ task runner for processing batch data or offline
 * streaming data. This method is designed for processing either batch data such as unrelated images
 * and texts or offline streaming data such as the decoded frames from a video file or audio file.
 * The call blocks the current thread until a failure status or a successful result is returned. If
 * the input packets have no timestamp, an internal timestamp will be assigned per invocation.
 * Otherwise, when the timestamp is set in the input packets, the caller must ensure that the input
 * packet timestamps are greater than the timestamps of the previous invocation. This method is
 * thread-unsafe and it is the caller's responsibility to synchronize access to this method across
 * multiple threads and to ensure that the input packet timestamps are in order.
 *
 * @param packetMap A `PacketMap` containing pairs of input stream name and data packet which are to
 * be sent to the C++ task runner for processing synchronously.
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 * error will be saved.
 *
 * @return An optional output `PacketMap` containing pairs of output stream name and data packet
 * which holds the results of processing the input packet map, if there are no errors.
 */
- (std::optional<mediapipe::tasks::core::PacketMap>)
    processPacketMap:(const mediapipe::tasks::core::PacketMap &)packetMap
               error:(NSError **)error;

/**
 * An asynchronous method that is designed for handling live streaming data such as live camera. A
 * user-defined PacketsCallback function must be provided in the constructor to receive the output
 * packets. The caller must ensure that the input packet timestamps are monotonically increasing.
 * This method is thread-unsafe and it is the caller's responsibility to synchronize access to this
 * method across multiple threads and to ensure that the input packet timestamps are in order.
 *
 * @param packetMap A `PacketMap` containing pairs of input stream name and data packet that are to
 * be sent to the C++ task runner for processing asynchronously.
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 * error will be saved.
 *
 * @return A `BOOL` indicating if the live stream data was sent to the C++ task runner successfully.
 * Please note that any errors during processing of the live stream packet map will only be
 * available in the user-defined `packetsCallback` that was provided during initialization of the
 * `MPPVisionTaskRunner`.
 */
- (BOOL)sendPacketMap:(const mediapipe::tasks::core::PacketMap &)packetMap error:(NSError **)error;

/**
 * Shuts down the C++ task runner. After the runner is closed, any calls that send input data to the
 * runner are illegal and will receive errors.
 *
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 * error will be saved.
 *
 * @return A `BOOL` indicating if the C++ task runner was shutdown successfully.
 */
- (BOOL)closeWithError:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
