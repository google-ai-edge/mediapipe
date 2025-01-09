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
#import <XCTest/XCTest.h>

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioRecord.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/test/audio/core/utils/sources/AVAudioPCMBuffer+TestUtils.h"
#import "mediapipe/tasks/ios/test/utils/sources/MPPFileInfo.h"

#include <algorithm>

static MPPFileInfo *const kSpeech16KHzMonoFileInfo =
    [[MPPFileInfo alloc] initWithName:@"speech_16000_hz_mono" type:@"wav"];

static MPPFileInfo *const kSpeech48KHzMonoFileInfo =
    [[MPPFileInfo alloc] initWithName:@"speech_48000_hz_mono" type:@"wav"];

static AVAudioFormat *const kAudioEngineFormat =
    [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                     sampleRate:48000.0f
                                       channels:1
                                    interleaved:YES];

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

NS_ASSUME_NONNULL_BEGIN

@interface MPPAudioRecordTests : XCTestCase
@end

// Equivalent of `@testable import` in Swift.
// Exposes the internal methods of `MPPAudioRecord` that are to be tested. This category is private
// to the current test file.
@interface MPPAudioRecord (Tests)

// -[MPPAudioRecord startRecordingWithError:] cannot be used for reproducible tests of the buffer
// loading mechanism of `MPPAudioRecord` since it records the input from the microphone. Instead the
// methods that perform the conversion to the format of the audio record and load the buffer are
// exposed here so that they can be used to load `wav` files. Together these methods mimic the
// loading of the audio record from the periodically available microphone samples.
+ (AVAudioPCMBuffer *)bufferFromInputBuffer:(AVAudioPCMBuffer *)pcmBuffer
                        usingAudioConverter:(AVAudioConverter *)audioConverter
                                      error:(NSError **)error;

- (BOOL)loadAudioPCMBuffer:(AVAudioPCMBuffer *)pcmBuffer error:(NSError **)error;

- (nullable MPPFloatBuffer *)internalReadAtOffset:(NSUInteger)offset
                                       withLength:(NSUInteger)length
                                            error:(NSError **)error;
@end

@implementation MPPAudioRecordTests

#pragma mark Tests

- (void)testInitAudioRecordFailsWithInvalidChannelCount {
  const NSInteger channelCount = 3;
  const double sampleRate = 8000.0f;
  MPPAudioDataFormat *audioDataFormat =
      [[MPPAudioDataFormat alloc] initWithChannelCount:channelCount sampleRate:sampleRate];

  NSError *error = nil;
  const NSInteger bufferLength = 100;
  MPPAudioRecord *audioRecord = [[MPPAudioRecord alloc] initWithAudioDataFormat:audioDataFormat
                                                                   bufferLength:bufferLength
                                                                          error:&error];
  XCTAssertNil(audioRecord);

  NSError *expectedError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey : @"The channel count provided does not match the "
                                           @"supported channel count. Only channels counts "
                                           @"in the range [1 : 2] are supported"
             }];

  AssertEqualErrors(error, expectedError);
}

- (void)testInitAudioRecordFailsWithInvalidBufferLength {
  const NSInteger channelCount = 2;
  const double sampleRate = 8000.0f;
  MPPAudioDataFormat *audioDataFormat =
      [[MPPAudioDataFormat alloc] initWithChannelCount:channelCount sampleRate:sampleRate];

  NSError *error = nil;
  const NSInteger bufferLength = 101;
  MPPAudioRecord *audioRecord = [[MPPAudioRecord alloc] initWithAudioDataFormat:audioDataFormat
                                                                   bufferLength:bufferLength
                                                                          error:&error];
  XCTAssertNil(audioRecord);

  NSError *expectedError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey :
                            [NSString stringWithFormat:@"The buffer length provided (%lu) is not a "
                                                       @"multiple of channel count(%lu).",
                                                       bufferLength, audioDataFormat.channelCount]
                      }];

  AssertEqualErrors(error, expectedError);
}

- (void)testConvertAndLoadAudioRecordWithMonoFormatSucceeds {
  const NSInteger channelCount = 1;
  const double sampleRate = 16000.0f;
  const NSInteger bufferLength = 100;

  [MPPAudioRecordTests
      assertCreateAndLoadAudioRecordSucceedsWithAudioFileInfo:kSpeech16KHzMonoFileInfo
                                                   sampleRate:sampleRate
                                                 bufferLength:bufferLength
                                                 channelCount:channelCount];
}

- (void)testConvertAndLoadAudioRecordWithStereoFormatSucceeds {
  const NSInteger channelCount = 2;
  const double sampleRate = 8000.0f;
  const NSInteger bufferLength = 200;
  [MPPAudioRecordTests
      assertCreateAndLoadAudioRecordSucceedsWithAudioFileInfo:kSpeech16KHzMonoFileInfo
                                                   sampleRate:sampleRate
                                                 bufferLength:bufferLength
                                                 channelCount:channelCount];
}

- (void)testConvertAndLoadAudioRecordWithMonoFormatAndLongBufferLengthSucceeds {
  const NSUInteger channelCount = 1;
  const NSUInteger sampleRate = 8000;

  // Buffer length is equal to the interim buffer produced by `MPPAudioRecord` after coversion of
  // the audio samples to its format. Test ensures that the entire buffer is loaded into the ring
  // buffer of audio record in this scenario.
  const NSUInteger expectedBufferLengthOfInternalConvertedAudioBuffer = 34180;

  // Buffer length is equal to the interim buffer produced by `MPPAudioRecord` after coversion of
  // the audio samples to its format. Test ensures that the entire buffer is loaded into the ring
  // buffer of audio record in this scenario.
  [MPPAudioRecordTests
      assertCreateAndLoadAudioRecordSucceedsWithAudioFileInfo:kSpeech16KHzMonoFileInfo
                                                   sampleRate:sampleRate
                                                 bufferLength:
                                                     expectedBufferLengthOfInternalConvertedAudioBuffer
                                                 channelCount:channelCount];

  // Buffer length is longer than the interim buffer produced by `MPPAudioRecord` after coversion of
  // the audio samples to its format. Test ensures that the entire buffer is loaded as the most
  // recent samples of the ring buffer of audio record by pushing out the oldest samples in this
  // scenario. After loading the ring buffer, the earliest samples should be all zeroes since we
  // start with a fresh audio record.
  const NSUInteger bufferLength = 40000;

  [MPPAudioRecordTests
      assertCreateAndLoadAudioRecordSucceedsWithAudioFileInfo:kSpeech16KHzMonoFileInfo
                                                   sampleRate:sampleRate
                                                 bufferLength:bufferLength
                                                 channelCount:channelCount];
}

- (void)testConvertAndRepeatedlyLoadAudioRecordWithMonoFormatSucceeds {
  const NSUInteger channelCount = 1;
  const NSUInteger sampleRate = 8000;

  // Buffer length is longer than the interim buffer produced by `MPPAudioRecord` after coversion of
  // the audio samples to its format. Test ensures that the entire buffer is loaded as the most
  // recent samples of the ring buffer of audio record by pushing out the oldest samples in this
  // scenario. After loading the ring buffer, the earliest samples should be all zeroes since we
  // start with a fresh audio record.
  const NSUInteger bufferLength = 40000;

  MPPAudioRecord *audioRecord = [MPPAudioRecordTests
      assertCreateAndLoadAudioRecordSucceedsWithAudioFileInfo:kSpeech16KHzMonoFileInfo
                                                   sampleRate:sampleRate
                                                 bufferLength:bufferLength
                                                 channelCount:channelCount];

  // Loads audio record with a second audio file to verify that older samples are shifted out to
  // make room for new samples when th audio record is already loaded.
  AVAudioPCMBuffer *bufferInAudioRecordFormat =
      [MPPAudioRecordTests bufferFromAudioFileWithInfo:kSpeech16KHzMonoFileInfo
                               withFormatOfAudioRecord:audioRecord];
  [MPPAudioRecordTests assertSuccessOfLoadAudioRecord:audioRecord
                                        withPCMBuffer:bufferInAudioRecordFormat];
}

- (void)testReadAudioRecordAtOffsetSucceeds {
  const NSUInteger channelCount = 1;
  const NSUInteger sampleRate = 8000;

  // Buffer length is equal to the interim buffer produced by `MPPAudioRecord` after coversion of
  // the audio samples to its format. Test ensures that the entire buffer is loaded into the ring
  // buffer of audio record in this scenario.
  const NSUInteger expectedBufferLengthOfInternalConvertedAudioBuffer = 34180;

  // Buffer length is equal to the interim buffer produced by `MPPAudioRecord` after coversion of
  // the audio samples to its format. Test ensures that the entire buffer is loaded into the ring
  // buffer of audio record in this scenario.
  MPPAudioRecord *audioRecord = [MPPAudioRecordTests
      assertCreateAndLoadAudioRecordSucceedsWithAudioFileInfo:kSpeech16KHzMonoFileInfo
                                                   sampleRate:sampleRate
                                                 bufferLength:
                                                     expectedBufferLengthOfInternalConvertedAudioBuffer
                                                 channelCount:channelCount];

  const NSUInteger offset = 4;
  const NSUInteger length = 4000;
  [MPPAudioRecordTests assertSuccessOfReadAudioRecord:audioRecord atOffset:offset length:length];
}

- (void)testReadAudioRecordAtOffsetFailsWithIndexOutOfBounds {
  const NSInteger channelCount = 1;
  const double sampleRate = 16000.0f;
  const NSInteger bufferLength = 100;

  MPPAudioRecord *audioRecord = [MPPAudioRecordTests
      assertCreateAndLoadAudioRecordSucceedsWithAudioFileInfo:kSpeech16KHzMonoFileInfo
                                                   sampleRate:sampleRate
                                                 bufferLength:bufferLength
                                                 channelCount:channelCount];

  const NSUInteger offset = 4;
  const NSUInteger length = 100;
  NSError *error;
  [audioRecord internalReadAtOffset:offset withLength:length error:&error];

  NSError *expectedError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey :
                   [NSString stringWithFormat:
                                 @"Index out of range. `offset` (%lu) + `length` (%lu) must be <= "
                                 @"`length` (%lu)",
                                 offset, length, audioRecord.bufferLength]
             }];

  AssertEqualErrors(error, expectedError);
}

#pragma mark Create and Load Audio Record Assertions

+ (MPPAudioRecord *)assertCreateAndLoadAudioRecordSucceedsWithAudioFileInfo:(MPPFileInfo *)fileInfo
                                                                 sampleRate:(NSUInteger)sampleRate
                                                               bufferLength:(NSUInteger)bufferLength
                                                               channelCount:
                                                                   (NSUInteger)channelCount {
  MPPAudioRecord *audioRecord =
      [MPPAudioRecordTests createAudioRecordWithChannelCount:channelCount
                                                  sampleRate:sampleRate
                                                bufferLength:bufferLength];

  AVAudioPCMBuffer *bufferInAudioRecordFormat =
      [MPPAudioRecordTests bufferFromAudioFileWithInfo:kSpeech48KHzMonoFileInfo
                               withFormatOfAudioRecord:audioRecord];

  [MPPAudioRecordTests assertSuccessOfLoadAudioRecord:audioRecord
                                        withPCMBuffer:bufferInAudioRecordFormat];

  return audioRecord;
}

+ (MPPAudioRecord *)createAudioRecordWithChannelCount:(const NSInteger)channelCount
                                           sampleRate:(const NSInteger)sampleRate
                                         bufferLength:(const NSInteger)bufferLength {
  MPPAudioDataFormat *audioDataFormat =
      [[MPPAudioDataFormat alloc] initWithChannelCount:channelCount sampleRate:sampleRate];

  MPPAudioRecord *audioRecord = [[MPPAudioRecord alloc] initWithAudioDataFormat:audioDataFormat
                                                                   bufferLength:bufferLength
                                                                          error:nil];
  XCTAssertNotNil(audioRecord);

  return audioRecord;
}

+ (AVAudioPCMBuffer *)bufferFromAudioFileWithInfo:(MPPFileInfo *)fileInfo
                          withFormatOfAudioRecord:(MPPAudioRecord *)audioRecord {
  // Loading `AVAudioPCMBuffer` with an array is not currently supported for iOS versions < 15.0.
  // Instead audio samples from a wav file are loaded and converted into the same format
  // of `AVAudioEngine`'s input node to mock the input from the `AVAudioEngine`.
  AVAudioPCMBuffer *audioEngineBuffer =
      [AVAudioPCMBuffer bufferFromAudioFileWithInfo:fileInfo processingFormat:kAudioEngineFormat];
  XCTAssertNotNil(audioEngineBuffer);

  AVAudioFormat *recordingFormat =
      [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                       sampleRate:audioRecord.audioDataFormat.sampleRate
                                         channels:audioRecord.audioDataFormat.channelCount
                                      interleaved:YES];

  AVAudioConverter *audioConverter = [[AVAudioConverter alloc] initFromFormat:kAudioEngineFormat
                                                                     toFormat:recordingFormat];
  XCTAssertNotNil(audioConverter);

  // Load the audio record with the input buffer from the mock AVAudioEngine.
  AVAudioPCMBuffer *bufferInRecordingFormat =
      [MPPAudioRecord bufferFromInputBuffer:audioEngineBuffer
                        usingAudioConverter:audioConverter
                                      error:nil];
  XCTAssertNotNil(bufferInRecordingFormat);

  return bufferInRecordingFormat;
}

+ (void)assertSuccessOfLoadAudioRecord:(MPPAudioRecord *)audioRecord
                         withPCMBuffer:(AVAudioPCMBuffer *)bufferInAudioRecordFormat {
  MPPFloatBuffer *previousAudioRecordBuffer =
      [audioRecord internalReadAtOffset:0 withLength:audioRecord.bufferLength error:nil];

  XCTAssertTrue([audioRecord loadAudioPCMBuffer:bufferInAudioRecordFormat error:nil]);

  MPPFloatBuffer *audioRecordBuffer =
      [MPPAudioRecordTests readFullLengthBufferOfAudioRecord:audioRecord];
  [MPPAudioRecordTests assertFloatBuffer:audioRecordBuffer
      containsInOrderSamplesFromPreviousStateOfFloatBuffer:previousAudioRecordBuffer
                       containsInOrderSamplesFromPCMBuffer:bufferInAudioRecordFormat];
}

#pragma mark Read AudioRecord Assertion

+ (void)assertSuccessOfReadAudioRecord:(MPPAudioRecord *)audioRecord
                              atOffset:(NSUInteger)offset
                                length:(NSUInteger)length {
  MPPFloatBuffer *floatBuffer = [MPPAudioRecordTests readAudioRecord:audioRecord
                                                            atOffset:offset
                                                              length:length];
  MPPFloatBuffer *fullLengthAudioRecordFloatBuffer =
      [MPPAudioRecordTests readFullLengthBufferOfAudioRecord:audioRecord];
  for (int i = 0; i < floatBuffer.length; i++) {
    XCTAssertEqualWithAccuracy(floatBuffer.data[i],
                               fullLengthAudioRecordFloatBuffer.data[offset + i], FLT_EPSILON);
  }
}

+ (MPPFloatBuffer *)readFullLengthBufferOfAudioRecord:(MPPAudioRecord *)audioRecord {
  return [MPPAudioRecordTests readAudioRecord:audioRecord
                                     atOffset:0
                                       length:audioRecord.bufferLength];
}

+ (MPPFloatBuffer *)readAudioRecord:(MPPAudioRecord *)audioRecord
                           atOffset:(NSUInteger)offset
                             length:(NSUInteger)length {
  MPPFloatBuffer *audioRecordBuffer = [audioRecord internalReadAtOffset:offset
                                                             withLength:length
                                                                  error:nil];
  XCTAssertNotNil(audioRecordBuffer);
  XCTAssertEqual(audioRecordBuffer.length, length);
  return audioRecordBuffer;
}

#pragma mark Helper Assertions

// Verifies if the current float buffer is created by shifting out old samples to make room for
// samples from the `pcmBuffer`.
+ (void)assertFloatBuffer:(MPPFloatBuffer *)floatBuffer
    containsInOrderSamplesFromPreviousStateOfFloatBuffer:
        (MPPFloatBuffer *)previousStateOfFloatBuffer
                     containsInOrderSamplesFromPCMBuffer:(AVAudioPCMBuffer *)pcmBuffer {
  // The float buffer read from `MPPAudioRecord` is compared with samples in the `pcmBuffer`.
  // The float buffer is compared in 2 chunks. If the float buffer is shorter than the pcmBuffer,
  // the len(first chunk) = 0 and len(second chunk) = full length of float buffer i.e, the entire
  // float buffer is compared with the most recent samples of length = float buffer length in the
  // `pcmBuffer`. If the float buffer is longer than the `pcmBuffer`, the first chunk (oldest
  // samples) must be equal to its previous state offset from the len(pcmBuffer) to verify that
  // older elements are shifted out to make room for new elements from the `pcmBuffer`. len(second
  // chunk) = len(`pcmBuffer`). All samples of `pcmBuffer` must replace the samples in the second
  // chunk of the float buffer.
  const NSInteger secondChunkLength =
      std::min((NSInteger)floatBuffer.length, (NSInteger)pcmBuffer.frameLength);
  const NSInteger firstChunkLength = floatBuffer.length - secondChunkLength;

  for (int i = 0; i < firstChunkLength; i++) {
    XCTAssertEqualWithAccuracy(floatBuffer.data[i],
                               previousStateOfFloatBuffer.data[i + pcmBuffer.frameLength],
                               FLT_EPSILON);
  }

  // Starting indices for comparison of the second chunks in float bufer and `pcmBuffer` are
  // calculated.
  const NSInteger startIndexForComparisonInFloatBuffer = firstChunkLength;
  const NSInteger startIndexForComparisonInPCMBuffer = pcmBuffer.frameLength - secondChunkLength;

  for (int i = 0; i < secondChunkLength; i++) {
    XCTAssertEqualWithAccuracy(
        floatBuffer.data[startIndexForComparisonInFloatBuffer + i],
        pcmBuffer.floatChannelData[0][startIndexForComparisonInPCMBuffer + i], FLT_EPSILON);
  }
}

@end

NS_ASSUME_NONNULL_END
