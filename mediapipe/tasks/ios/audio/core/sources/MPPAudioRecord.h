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

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioDataFormat.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPFloatBuffer.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * A wrapper class to record the device's microphone continuously. Currently this class only
 * supports recording up to 2 channels. If the number of channels is 2, then the mono microphone
 * input is duplicated to provide dual channel data.
 *
 * Internally the class manages an instance of `AVAudioEngine` for tapping the microphone samples.
 *
 * Recording sets the following parameters on the `AVAudioSession.sharedInstance()`:
 *
 * 1. `category` = .playAndRecord
 * 2. `mode` = .default
 * 3. `categoryOptions` = [.overrideMutedMicrophoneInterruption]
 *
 * If you need more control, use the native `AVAudioEngine` directly for tapping the microphone.
 * `AudioRecord` does not actively monitor audio interruptions, route changes, resetting of media
 * services etc. They can be handled in the app using the notifications provided by `AVFAudio`.
 * Refer to the official docs for the following:
 * 1. Handling audio interruptions
 * https://developer.apple.com/documentation/avfaudio/handling_audio_interruptions
 *
 * 2. Responding to audio route changes
 * https://developer.apple.com/documentation/avfaudio/responding_to_audio_route_change
 *
 * 3. Responding to resetting of media services
 * https://developer.apple.com/documentation/avfaudio/avaudiosessionmediaserviceswereresetnotification
 *
 * iOS may automatically stop the audio engine in some cases. A few examples of such events are a
 * system interrupt, route change, etc. In such cases, `read(offset:length:)` returns
 * `TasksErrorCode.audioRecordNotTappingMicError`. You can restart recording using
 * `startRecording()`.
 *
 * Note: Avoid setting the `AVAudioSession.sharedInstance()` category, mode, options, channel
 * configuration (enabling stereo) or other parameters while audio is being actively recorded using
 * the audio record. Setting these parameters or changing the configuration of
 * `AVAudioSession.sharedInstance()` while a recording is in progress will result in undefined
 * behaviour.
 */
NS_SWIFT_NAME(AudioRecord)
@interface MPPAudioRecord : NSObject

/** Audio format specifying the number of channels and sample rate supported. */
@property(nonatomic, readonly) MPPAudioDataFormat *audioDataFormat;

/**
 * Size of the buffer held by `AudioRecord`. It ensures delivery of audio  data of length
 * `bufferLength` arrays when you start recording the microphone input.
 */
@property(nonatomic, readonly) NSUInteger bufferLength;

/**
 * Initializes a new `AudioRecord` with the given audio format and buffer length.
 *
 * @param format An audio format of type `AudioDataFormat`.
 * @param bufferLength Maximum number of elements the internal buffer of `AudioRecord` can hold at
 * any given point of time. The buffer length must be a multiple of `format.channelCount`.
 *
 * @return An new instance of `AudioRecord` with the given audio format and buffer length. `nil` if
 * there is an error in initializing `AudioRecord`.
 */
- (nullable instancetype)initWithAudioDataFormat:(MPPAudioDataFormat *)format
                                    bufferLength:(NSUInteger)bufferSize
                                           error:(NSError **)error;

/**
 * This function starts recording the audio from the microphone if audio record permissions
 * have been granted by the user.
 *
 * Before calling this function, you must use
 * `AVAudioSession.sharedInstance.requestRecordPermission()` to acquire record permissions. If the
 * user has denied permission or the permissions are undetermined, this method will throw an error.
 * The internal buffer of `AudioRecord` of will always have the most recent audio samples acquired
 * from the microphhone if this function returns successfully.  Use: `read(offset:length:)` to get
 * the data from the buffer at any point of time after audio recording has started successfully.
 *
 * Use `stop()` to stop recording audio sample from the microphone.
 *
 * Internally this function activates the `AVAudioSession.sharedInstance()` with the following
 * parameters before tapping the microphone using the `AVAudioEngine`:
 * 1. `category` = .playAndRecord
 * 2. `mode` = .default
 * 3. `categoryOptions` = [.overrideMutedMicrophoneInterruption]
 *
 * If you need more control, use the native `AVAudioEngine` directly for tapping the microphone.
 * `AudioRecord` does not actively monitor audio interruptions, route changes, resetting of media
 * services etc. They can be handled in the app using the notifications provided by `AVFAudio`.
 * Refer to the official docs for the following:
 * 1. Handling audio interruptions
 * https://developer.apple.com/documentation/avfaudio/handling_audio_interruptions
 *
 * 2. Responding to audio route changes
 * https://developer.apple.com/documentation/avfaudio/responding_to_audio_route_change
 *
 * 3. Responding to resetting of media services
 * https://developer.apple.com/documentation/avfaudio/avaudiosessionmediaserviceswereresetnotification
 *
 * @return Returns successfully if audio recording has started.
 */
- (BOOL)startRecordingWithError:(NSError **)error NS_SWIFT_NAME(startRecording());

/**
 * Stops recording audio from the microphone. All elements in the internal buffer of `AudioRecord`
 * will also be set to zero.
 *
 * Internally, this function deactivates the `AVAudioSession.sharedInstance()` after recording is
 * stopped. If some audio resources are running in the app when the session is deactivated, this
 * function throws an error. The session gets deactivated by the OS irrespective of the return
 * status of this function. The error thrown is a warning to indicate the termination of any running
 * audio resources. Refer to the official documentation for more details:
 * https://developer.apple.com/documentation/avfaudio/avaudiosession/1616627-setactive
 *
 * @return Returns successfully if recording was stopped and the deactivation of
 * `AVAudioSession.sharedInstance()` succeeded. Throws any error encountered when deactivating
 * `AVAudioSession.sharedInstance()`.
 */
- (BOOL)stopWithError:(NSError **)error;

/**
 * Returns the `length` number of elements in the internal buffer of `AudioRecord` starting at
 * `offset`, i.e, `buffer[offset:offset+length]`.
 *
 * iOS may automatically stop the audio engine in some cases. A few examples of such events are a
 * system interrupt, route change, etc. In such cases, this function returns
 * `TasksErrorCode.audioRecordNotTappingMicError`. You can restart recording using
 * `startRecording()`.
 *
 * @param offset Index in the buffer from which elements are to be read.
 * @param length Number of elements to be returned.
 *
 * @return A `FloatBuffer` containing the elements of the internal buffer of `AudioRecord` in
 * the range, `buffer[offset:offset+size]`. Returns `nil` if there is an error in reading the
 * internal buffer.
 */
- (nullable MPPFloatBuffer *)readAtOffset:(NSUInteger)offset
                               withLength:(NSUInteger)length
                                    error:(NSError **)error NS_SWIFT_NAME(read(offset:length:));

@end

NS_ASSUME_NONNULL_END
