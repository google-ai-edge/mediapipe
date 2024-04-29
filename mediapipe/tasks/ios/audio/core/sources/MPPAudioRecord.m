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

#import <AVFoundation/AVFoundation.h>

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioRecord.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPFloatRingBuffer.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

static const NSUInteger kMaximumChannelCount = 2;

@implementation MPPAudioRecord {
  AVAudioEngine *_audioEngine;

  /**
   * Specifying a custom buffer size on `AVAudioEngine` while tapping audio does not take effect.
   * Hence we store the returned samples in a ring buffer to acheive the desired buffer length. If
   * the specified buffer length is shorter than the buffer length supported by `AVAudioEngine` only
   * the most recent data of the buffer of length `bufferLength` will be stored by the ring buffer.
   */
  MPPFloatRingBuffer *_floatRingBuffer;

  /**
   * Stores any error during buffer conversion or initial startup of the audio engine. Updated to
   * `nil` whenever a valid buffer entry is made to the internal ring buffer. This will be returned
   * when user tries to access an invalid buffer using `[MPPAudioRecord readAtOffset:length:error]`.
   * Writes and reads are restricted to the same queue used to access the internal ring buffer.
   */
  NSError *_globalError;

  /**
   * Concurrent queue that permits multiple threads to read the ring buffer and the global error
   * simultaneously but only allows a single thread to write to these instance variables. During
   * write, all read operations are blocked.
   *
   * TODO: Test if moving conversion to a dedicated queue offers any advantages in terms of
   * performance.
   */
  dispatch_queue_t _convertLoadAndReadBufferQueue;
}

- (nullable instancetype)initWithAudioDataFormat:(MPPAudioDataFormat *)audioDataFormat
                                    bufferLength:(NSUInteger)bufferLength
                                           error:(NSError **)error {
  self = [super init];
  if (self) {
    if (audioDataFormat.channelCount > kMaximumChannelCount || audioDataFormat.channelCount == 0) {
      [MPPCommonUtils
          createCustomError:error
                   withCode:MPPTasksErrorCodeInvalidArgumentError
                description:[NSString
                                stringWithFormat:@"The channel count provided does not match the "
                                                 @"supported channel count. Only channels counts "
                                                 @"in the range [1 : %lu] are supported",
                                                 kMaximumChannelCount]];
      return nil;
    }

    if (bufferLength % audioDataFormat.channelCount != 0) {
      [MPPCommonUtils
          createCustomError:error
                   withCode:MPPTasksErrorCodeInvalidArgumentError
                description:[NSString stringWithFormat:@"The buffer length provided (%lu) is not a "
                                                       @"multiple of channel count(%lu).",
                                                       bufferLength, audioDataFormat.channelCount]];
      return nil;
    }

    _audioDataFormat = audioDataFormat;
    _audioEngine = [[AVAudioEngine alloc] init];
    _bufferLength = bufferLength;

    _floatRingBuffer = [[MPPFloatRingBuffer alloc] initWithLength:_bufferLength];

    // Concurrent queue which permits multiple threads to read the  audio record simultaneously.
    // Writes are restricted to only one thread during which no reads are permitted.
    _convertLoadAndReadBufferQueue =
        dispatch_queue_create("org.mediapipe.AudioRecordReadWriteQueue", DISPATCH_QUEUE_CONCURRENT);
  }
  return self;
}

- (BOOL)startRecordingWithError:(NSError **)error {
  // TODO: This API is deprecated from iOS 17.0. Update to new APIs and restrict the following
  // code's use to versions below iOS 17.0.
  switch ([AVAudioSession sharedInstance].recordPermission) {
    case AVAudioSessionRecordPermissionDenied: {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeAudioRecordPermissionDeniedError
                            description:@"Record permission was denied by the user. "];
      break;
    }
    case AVAudioSessionRecordPermissionUndetermined: {
      [MPPCommonUtils
          createCustomError:error
                   withCode:MPPTasksErrorCodeAudioRecordPermissionUndeterminedError
                description:@"Record permissions are undertermined. Yo must use AVAudioSession's "
                            @"requestRecordPermission() to request audio record permission from "
                            @"the user. Please read Apple's documentation for further details"
                            @"If record permissions are granted, you can call this "
                            @"method in the completion handler of requestRecordPermission()."];
      break;
    }

    case AVAudioSessionRecordPermissionGranted: {
      [self startTappingMicrophoneWithError:error];
      return YES;
    }
  }

  return NO;
}

- (void)stop {
  [[_audioEngine inputNode] removeTapOnBus:0];
  [_audioEngine stop];

  // Using strong `self` (instance variable is available through strong self) is okay since the
  // block is shortlived and it'll release its strong reference to `self` when it finishes
  // execution.
  //
  // `dispatch_barrier_async` ensures no other thread can access `_floatRingBuffer` for read and
  // write while buffer is being cleared.
  dispatch_barrier_async(_convertLoadAndReadBufferQueue, ^{
    [_floatRingBuffer clear];
  });
}

- (nullable MPPFloatBuffer *)readAtOffset:(NSUInteger)offset
                               withLength:(NSUInteger)length
                                    error:(NSError **)error {
  __block MPPFloatBuffer *bufferToReturn = nil;
  __block NSError *readError = nil;

  // Using strong `self` (instance variable is available through strong self)  is okay since block
  // is shortlived and it'll release its strong reference to `self` when it finishes execution. This
  // allows multiple threads to read the buffer at the same time. `sync` ensures the method
  // execution (return) is blocked until the read is completed.
  dispatch_sync(_convertLoadAndReadBufferQueue, ^{
    if (!_globalError) {
      bufferToReturn = [_floatRingBuffer floatBufferWithOffset:offset
                                                        length:length
                                                         error:&readError];
    } else {
      readError = [_globalError copy];
    }
  });

  if (readError) {
    *error = readError;
  }

  return bufferToReturn;
}

- (void)startTappingMicrophoneWithError:(NSError **)error {
  AVAudioNode *inputNode = [_audioEngine inputNode];
  AVAudioFormat *format = [inputNode outputFormatForBus:0];

  AVAudioFormat *recordingFormat = [[AVAudioFormat alloc]
      initWithCommonFormat:AVAudioPCMFormatFloat32
                sampleRate:self.audioDataFormat.sampleRate
                  channels:(AVAudioChannelCount)self.audioDataFormat.channelCount
               interleaved:YES];

  AVAudioConverter *audioConverter = [[AVAudioConverter alloc] initFromFormat:format
                                                                     toFormat:recordingFormat];

  // Making self weak for the `installTapOnBus` callback.
  __weak MPPAudioRecord *weakSelf = self;

  // Setting buffer size takes no effect on the input node. This class uses a ring buffer internally
  // to ensure the requested buffer size.
  [inputNode installTapOnBus:0
                  bufferSize:(AVAudioFrameCount)self.bufferLength
                      format:format
                       block:^(AVAudioPCMBuffer *buffer, AVAudioTime *when) {
                         //  Getting a strong reference to `weakSelf` to conditionally execute
                         //  conversion and ring buffer loading. If self is deallocated before the
                         //  block is called, then `strongSelf` will be `nil`. Thereafter it is kept
                         //  in memory until the block finishes execution.
                         __strong MPPAudioRecord *strongSelf = weakSelf;
                         // Check for non NULL here since argument of `dispatch_sync` cannot be
                         // NULL. Conversion and writing to buffer is done with
                         // `dispatch_barrier_async` to ensure that no other thread has access to
                         // the ring buffer and global error until the write is completed.
                         if (strongSelf) {
                           dispatch_barrier_async(strongSelf->_convertLoadAndReadBufferQueue, ^{
                             NSError *convertAndLoadError = nil;
                             [strongSelf convertAndLoadBuffer:buffer
                                          usingAudioConverter:audioConverter
                                                        error:&convertAndLoadError];
                             strongSelf->_globalError = convertAndLoadError;
                           });
                         }
                       }];

  [_audioEngine prepare];
  [_audioEngine startAndReturnError:error];
}

- (BOOL)loadAudioPCMBuffer:(AVAudioPCMBuffer *)pcmBuffer error:(NSError **)error {
  if (pcmBuffer.frameLength == 0) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"You may have to try with a different "
                                      @"channel count or sample rate"];
    return NO;
  }

  if (pcmBuffer.format.commonFormat != AVAudioPCMFormatFloat32) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Invalid pcm buffer format."];
    return NO;
  }

  // `pcmBuffer` is already converted to an interleaved format since this method is called after
  // -[self bufferFromInputBuffer:usingAudioConverter:error:].
  // If an `AVAudioPCMBuffer` is interleaved, both floatChannelData[0] and floatChannelData[1]
  // point to the same 1d array with both channels in an interleaved format according to:
  // https://developer.apple.com/documentation/avfaudio/avaudiopcmbuffer/1386212-floatchanneldata
  // Hence we can safely access floatChannelData[0] to get the 1D data in interleaved fashion.
  return [_floatRingBuffer
      loadFloatBuffer:[[MPPFloatBuffer alloc] initWithData:pcmBuffer.floatChannelData[0]
                                                    length:pcmBuffer.frameLength]
               offset:0
               length:pcmBuffer.frameLength
                error:error];
}

- (BOOL)convertAndLoadBuffer:(AVAudioPCMBuffer *)buffer
         usingAudioConverter:(AVAudioConverter *)audioConverter
                       error:(NSError **)error {
  AVAudioPCMBuffer *convertedPCMBuffer = [MPPAudioRecord bufferFromInputBuffer:buffer
                                                           usingAudioConverter:audioConverter
                                                                         error:error];

  return convertedPCMBuffer ? [self loadAudioPCMBuffer:convertedPCMBuffer error:error] : NO;
}

+ (AVAudioPCMBuffer *)bufferFromInputBuffer:(AVAudioPCMBuffer *)pcmBuffer
                        usingAudioConverter:(AVAudioConverter *)audioConverter
                                      error:(NSError **)error {
  // Capacity of converted PCM buffer is calculated in order to maintain the same latency as the
  // input pcmBuffer.
  AVAudioFrameCount capacity = ceil(pcmBuffer.frameLength * audioConverter.outputFormat.sampleRate /
                                    audioConverter.inputFormat.sampleRate);
  AVAudioPCMBuffer *outPCMBuffer = [[AVAudioPCMBuffer alloc]
      initWithPCMFormat:audioConverter.outputFormat
          frameCapacity:capacity * (AVAudioFrameCount)audioConverter.outputFormat.channelCount];

  AVAudioConverterInputBlock inputBlock = ^AVAudioBuffer *_Nullable(
      AVAudioPacketCount inNumberOfPackets, AVAudioConverterInputStatus *_Nonnull outStatus) {
    *outStatus = AVAudioConverterInputStatus_HaveData;
    return pcmBuffer;
  };

  NSError *conversionError = nil;
  AVAudioConverterOutputStatus converterStatus = [audioConverter convertToBuffer:outPCMBuffer
                                                                           error:&conversionError
                                                              withInputFromBlock:inputBlock];

  switch (converterStatus) {
    case AVAudioConverterOutputStatus_HaveData:
      return outPCMBuffer;
    case AVAudioConverterOutputStatus_Error: {
      NSString *errorDescription = conversionError.localizedDescription
                                       ? conversionError.localizedDescription
                                       : @"Some error occured while processing incoming audio "
                                         @"frames.";
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInternalError
                            description:errorDescription];
      break;
    }
    case AVAudioConverterOutputStatus_EndOfStream: {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInternalError
                            description:@"Reached end of input audio stream. "];
      break;
    }
    case AVAudioConverterOutputStatus_InputRanDry: {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInternalError
                            description:@"Not enough input is available to satisfy the request."];
      break;
    }
  }
  return nil;
}

@end
