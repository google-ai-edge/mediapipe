// Copyright 2024 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioData.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioRecord.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioRunningMode.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskRunner.h"

#include "mediapipe/framework/packet.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * This class is used to create and call appropriate methods on the C++ Task Runner to initialize,
 * execute and terminate any MediaPipe audio task.
 */
@interface MPPAudioTaskRunner : MPPTaskRunner

/**
 * Initializes a new `MPPAudioTaskRunner` with the given task info, audio running mode, packets
 * callback, audio input and sample rate stream names. Make sure that the packets callback is set
 * properly based on the audio task's running mode. In case of audio stream running mode, a C++
 * packets callback that is intended to deliver inference results must be provided. In audio clips
 * mode, packets callback must be set to nil.
 *
 * @param taskInfo A `MPPTaskInfo` initialized by the task.
 * @param runningMode MediaPipe audio task running mode.
 * @param packetsCallback An optional C++ callback function that takes a list of output packets as
 * the input argument. If provided, the callback must in turn call the block provided by the user in
 * the appropriate task options. Make sure that the packets callback is set properly based on the
 * audio task's running mode. In case of audio stream running mode, a C++ packets callback that is
 * intended to deliver inference results must be provided. In audio clips running mode, packets
 * callback must be set to nil.
 * @param audioInputStreamName Name of the audio input stream of the task.
 * @param sampleRatInputStreamName Name of the sample rate input stream of the task.
 *
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 * error will be saved.
 *
 * @return An instance of `MPPAudioTaskRunner` initialized with the given task info, running mode,
 * packets callback, audio input and sample rate stream names.
 */

- (nullable instancetype)initWithTaskInfo:(MPPTaskInfo *)taskInfo
                              runningMode:(MPPAudioRunningMode)runningMode
                          packetsCallback:(mediapipe::tasks::core::PacketsCallback)packetsCallback
                     audioInputStreamName:(NSString *)audioInputStreamName
                sampleRateInputStreamName:(nullable NSString *)normRectInputStreamName
                                    error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * A synchronous method to invoke the C++ task runner to process standalone audio clip inputs. The
 * call blocks the current thread until a failure status or a successful result is returned.
 *
 *
 * @param audioClip An audio clip input of type `MPPAudioData` to the task.
 * @param error Pointer to the memory location where errors if any should be
 * saved. If @c NULL, no error will be saved.
 *
 * @return An optional `PacketMap` containing pairs of output stream name and data packet.
 */
- (std::optional<mediapipe::tasks::core::PacketMap>)processAudioClip:(MPPAudioData *)audioClip
                                                               error:(NSError **)error;

/**
 * An asynchronous method to send audio stream data to the C++ task runner. The call returns
 * immediately indicating if the audio clip was sent successfully to the C++ task runner. The
 * results will be available in the user-defined `packetsCallback` that was provided during
 * initialization of the `MPPAudioTaskRunner`.
 *
 * @param audioClip An audio clip input of type `MPPAudioData` to the task.
 * @param timestampInMilliseconds The audio clip's timestamp (in milliseconds). The input timestamps
 * must be monotonically increasing.
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 * error will be saved.
 *
 * @return A `BOOL` indicating if the audio stream data was sent to the C++ task runner
 * successfully. Please note that any errors during processing of the audio stream packet map will
 * only be available in the user-defined `packetsCallback` that was provided during initialization
 * of the `MPPAudioTaskRunner`.
 */
- (BOOL)processStreamAudioClip:(MPPAudioData *)audioClip
       timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                         error:(NSError **)error;

/**
 * Creates an `MPPAudioRecord` instance to get samples from the audio stream produced by the
 * microphone.
 *
 * The client must call appropriate methods from the audio record to start receiving samples from
 * the microphone.
 *
 * Note that MediaPipe Audio tasks will up/down sample automatically to fit the sample rate required
 * by the model. The default sample rate of the MediaPipe pretrained audio model, Yamnet is 16kHz.
 *
 * @param channelCount Number of channels expected by the client.
 * @param sampleRate Sample rate of the audio expected by the client.
 * @param bufferLength Maximum number of elements the internal buffer of `AudioRecord` can hold at
 * any given point of time. The buffer length must be a multiple of `format.channelCount`.
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 * error will be saved.
 *
 * @return An new instance of `MPPAudioRecord` with the given audio format and buffer length. `nil`
 * if there is an error in initializing `MPPAudioRecord`.
 */
+ (MPPAudioRecord *)createAudioRecordWithChannelCount:(NSUInteger)channelCount
                                           sampleRate:(double)sampleRate
                                         bufferLength:(NSUInteger)bufferLength
                                                error:(NSError **)error;

- (instancetype)initWithTaskInfo:(MPPTaskInfo *)taskInfo
                 packetsCallback:(mediapipe::tasks::core::PacketsCallback)packetsCallback
                           error:(NSError **)error NS_UNAVAILABLE;

- (instancetype)init NS_UNAVAILABLE;

/**
 * This method returns a unique dispatch queue name by adding the given suffix and a `UUID` to the
 * pre-defined queue name prefix for audio tasks. The audio tasks can use this method to get
 * unique dispatch queue names which are consistent with other audio tasks.
 * Dispatch queue names need not be unique, but for easy debugging we ensure that the queue names
 * are unique.
 *
 * @param suffix A suffix that identifies a dispatch queue's functionality.
 *
 * @return A unique dispatch queue name by adding the given suffix and a `UUID` to the pre-defined
 * queue name prefix for audio tasks.
 */
+ (const char *)uniqueDispatchQueueNameWithSuffix:(NSString *)suffix;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
