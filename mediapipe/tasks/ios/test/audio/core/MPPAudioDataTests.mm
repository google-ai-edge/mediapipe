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

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioData.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/test/audio/core/utils/sources/AVAudioPCMBuffer+TestUtils.h"
#import "mediapipe/tasks/ios/test/utils/sources/MPPFileInfo.h"

#include <algorithm>

static MPPFileInfo *const kSpeech16KHzMonoFileInfo =
    [[MPPFileInfo alloc] initWithName:@"speech_16000_hz_mono" type:@"wav"];

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

#define AssertAudioDataIsValid(audioData, channelCount, sampleRate, sampleCount) \
  XCTAssertNotNil(audioData);                                                    \
  XCTAssertEqual(audioData.format.channelCount, channelCount);                   \
  XCTAssertEqual(audioData.format.sampleRate, sampleRate);                       \
  XCTAssertEqual(audioData.bufferLength, sampleCount);

#define AssertEqualFloatBuffers(buffer, expectedBuffer)     \
  XCTAssertNotNil(buffer);                                  \
  XCTAssertNotNil(expectedBuffer);                          \
  XCTAssertEqual(buffer.length, expectedBuffer.length);     \
  for (int i = 0; i < buffer.length; i++) {                 \
    XCTAssertEqual(buffer.data[i], expectedBuffer.data[i]); \
  }

NS_ASSUME_NONNULL_BEGIN

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

@interface MPPAudioData ()
- (BOOL)loadRingBufferWithAudioRecordBuffer:audioRecordBuffer error:(NSError **)error;
- (BOOL)isValidAudioRecordFormat:(MPPAudioDataFormat *)format error:(NSError **)error;
@end

@interface MPPAudioData (Tests)
- (BOOL)mockLoadAudioRecord:(MPPAudioRecord *)audioRecord error:(NSError **)error;
@end

@implementation MPPAudioData (Tests)
// Mocks the logic of `loadAudioRecord` for tests to avoid audio engine running state checks.
- (BOOL)mockLoadAudioRecord:(MPPAudioRecord *)audioRecord error:(NSError **)error {
  if (![self isValidAudioRecordFormat:audioRecord.audioDataFormat error:error]) {
    return NO;
  }

  // Invoking `internalReadAtOffset` instead of `readAtOffset` to avoid audio engine running state
  // checks.
  MPPFloatBuffer *audioRecordBuffer = [audioRecord internalReadAtOffset:0
                                                             withLength:audioRecord.bufferLength
                                                                  error:error];

  return [self loadRingBufferWithAudioRecordBuffer:audioRecordBuffer error:error];
}
@end

@interface MPPAudioDataTests : XCTestCase
@end

@implementation MPPAudioDataTests

- (void)testInitWithFormatAndSampleCountSucceeds {
  const NSInteger monoChannelCount = 1;
  const double sampleRate = 16000.0f;
  const NSInteger sampleCount = 1200;

  [MPPAudioDataTests asssertCreateAudioDataWithChannelCount:monoChannelCount
                                                 sampleRate:sampleRate
                                                sampleCount:sampleCount];

  const NSInteger stereoChannelCount = 2;

  [MPPAudioDataTests asssertCreateAudioDataWithChannelCount:stereoChannelCount
                                                 sampleRate:sampleRate
                                                sampleCount:sampleCount];
}

- (void)testLoadWithFloatBufferSucceeds {
  const NSInteger monoChannelCount = 1;
  const double sampleRate = 16000.0f;
  const NSInteger sampleCount = 7;

  MPPAudioData *audioData =
      [MPPAudioDataTests asssertCreateAudioDataWithChannelCount:monoChannelCount
                                                     sampleRate:sampleRate
                                                    sampleCount:sampleCount];

  // Load verifies that the input data is loaded as the most recent samples in audio data.
  float firstInputData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const NSUInteger firstInputDataLength = 5;

  // expected state = {0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}
  [MPPAudioDataTests
      assertLoadSucceedsOnAudioData:audioData
                      withInputData:&(firstInputData[0])inputDataLength:firstInputDataLength];

  // Load verifies that the oldest samples in audio data are pushed out and the input data is loaded
  // as the most recent samples.
  float secondInputData[] = {6.0f, 7.0f, 8.0f};
  const NSUInteger secondInputDataLength = 3;

  // expected state = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}
  [MPPAudioDataTests
      assertLoadSucceedsOnAudioData:audioData
                      withInputData:&(secondInputData[0])inputDataLength:secondInputDataLength];

  // Load verifies that most recent samples from an input buffer longer than the audio data buffer
  // replace all of its previous samples.
  float thirdInputData[] = {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  const NSUInteger thirdInputDataLength = 8;

  // expected state = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  [MPPAudioDataTests
      assertLoadSucceedsOnAudioData:audioData
                      withInputData:&(thirdInputData[0])inputDataLength:thirdInputDataLength];
}

- (void)testLoadWithFloatBufferAndOffsetSucceeds {
  const NSInteger monoChannelCount = 1;
  const double sampleRate = 16000.0f;
  const NSInteger sampleCount = 7;

  MPPAudioData *audioData =
      [MPPAudioDataTests asssertCreateAudioDataWithChannelCount:monoChannelCount
                                                     sampleRate:sampleRate
                                                    sampleCount:sampleCount];

  float inputData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const NSUInteger inputDataLength = 5;
  const NSUInteger offset = 2;
  const NSUInteger lengthToBeLoaded = 2;

  // expected state = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 4.0f};
  [MPPAudioDataTests assertLoadSucceedsOnAudioData:audioData
                                     withInputData:&(inputData[0])inputDataLength:inputDataLength
                                            offset:offset
                                  lengthToBeLoaded:lengthToBeLoaded];
}

- (void)testLoadWithLengthOutOfBoundsFails {
  const NSInteger monoChannelCount = 1;
  const double sampleRate = 16000.0f;
  const NSInteger sampleCount = 7;

  MPPAudioData *audioData =
      [MPPAudioDataTests asssertCreateAudioDataWithChannelCount:monoChannelCount
                                                     sampleRate:sampleRate
                                                    sampleCount:sampleCount];

  float inputData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const NSUInteger inputDataLength = 5;

  // Offset is within bounds of input data, length iexceeds [offset : input data length].
  const NSUInteger offset = 2;
  const NSUInteger lengthToBeLoaded = 5;

  [MPPAudioDataTests assertFailureOfLoadAudioData:audioData
                                         withData:&(inputData[0])ofLength:inputDataLength
                                           offset:offset
                                 lengthToBeLoaded:lengthToBeLoaded];
}

- (void)testLoadWithOffsetOutOfBoundsFails {
  const NSInteger monoChannelCount = 1;
  const double sampleRate = 16000.0f;
  const NSInteger sampleCount = 7;

  MPPAudioData *audioData =
      [MPPAudioDataTests asssertCreateAudioDataWithChannelCount:monoChannelCount
                                                     sampleRate:sampleRate
                                                    sampleCount:sampleCount];

  float inputData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const NSUInteger inputDataLength = 5;

  // Offset exceeds length of input data.
  const NSUInteger offset = 5;
  const NSUInteger lengthToBeLoaded = 1;

  [MPPAudioDataTests assertFailureOfLoadAudioData:audioData
                                         withData:&(inputData[0])ofLength:inputDataLength
                                           offset:offset
                                 lengthToBeLoaded:lengthToBeLoaded];
}

- (void)testLoadFromShorterAudioRecordSucceeds {
  const NSInteger monoChannelCount = 1;
  const double sampleRate = 16000.0f;
  const NSInteger sampleCount = 1000;

  const NSInteger bufferLength = 400;

  MPPAudioData *audioData =
      [MPPAudioDataTests asssertCreateAudioDataWithChannelCount:monoChannelCount
                                                     sampleRate:sampleRate
                                                    sampleCount:sampleCount];

  MPPAudioRecord *audioRecord =
      [MPPAudioDataTests createAndLoadAudioRecordWithAudioFileInfo:kSpeech16KHzMonoFileInfo
                                                        sampleRate:sampleRate
                                                      bufferLength:bufferLength
                                                      channelCount:monoChannelCount];

  // Load verifies that all samples of the audio record are loaded as the most recent samples of the
  // audio data.
  [MPPAudioDataTests assertSuccessOfLoadAudioData:audioData fromAudioRecord:audioRecord];

  // Load verifies that all older samples in audio data are shifted out to make room for all the
  // samples from the audio record in a circular fashion. Note: Same audio record is loaded twice to
  // mimic audio record being loaded repeatedly through time.
  [MPPAudioDataTests assertSuccessOfLoadAudioData:audioData fromAudioRecord:audioRecord];
}

- (void)testLoadFromLongerAudioRecordSucceeds {
  const NSInteger monoChannelCount = 1;
  const double sampleRate = 16000.0f;
  const NSInteger sampleCount = 400;

  const NSInteger bufferLength = 1000;

  MPPAudioData *audioData =
      [MPPAudioDataTests asssertCreateAudioDataWithChannelCount:monoChannelCount
                                                     sampleRate:sampleRate
                                                    sampleCount:sampleCount];

  MPPAudioRecord *audioRecord =
      [MPPAudioDataTests createAndLoadAudioRecordWithAudioFileInfo:kSpeech16KHzMonoFileInfo
                                                        sampleRate:sampleRate
                                                      bufferLength:bufferLength
                                                      channelCount:monoChannelCount];

  // Load verifies that most recent samples of the audio record replace all samples of audio data.
  [MPPAudioDataTests assertSuccessOfLoadAudioData:audioData fromAudioRecord:audioRecord];
}

#pragma mark Audio Record Helpers
+ (MPPAudioRecord *)createAndLoadAudioRecordWithAudioFileInfo:(MPPFileInfo *)fileInfo
                                                   sampleRate:(NSUInteger)sampleRate
                                                 bufferLength:(NSUInteger)bufferLength
                                                 channelCount:(NSUInteger)channelCount {
  MPPAudioRecord *audioRecord = [MPPAudioDataTests createAudioRecordWithChannelCount:channelCount
                                                                          sampleRate:sampleRate
                                                                        bufferLength:bufferLength];

  AVAudioPCMBuffer *bufferInAudioRecordFormat =
      [MPPAudioDataTests bufferFromAudioFileWithInfo:fileInfo withFormatOfAudioRecord:audioRecord];

  [audioRecord loadAudioPCMBuffer:bufferInAudioRecordFormat error:nil];

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

  return audioRecord;
}

+ (AVAudioPCMBuffer *)bufferFromAudioFileWithInfo:(MPPFileInfo *)fileInfo
                          withFormatOfAudioRecord:(MPPAudioRecord *)audioRecord {
  // Loading `AVAudioPCMBuffer` with an array is not currently supported for iOS versions < 15.0.
  // Instead audio samples from a wav file are loaded and converted into the same format
  // of `AVAudioEngine`'s input node to mock the input from the `AVAudioEngine`.
  AVAudioPCMBuffer *audioEngineBuffer =
      [AVAudioPCMBuffer bufferFromAudioFileWithInfo:fileInfo processingFormat:kAudioEngineFormat];

  AVAudioFormat *recordingFormat =
      [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                       sampleRate:audioRecord.audioDataFormat.sampleRate
                                         channels:audioRecord.audioDataFormat.channelCount
                                      interleaved:YES];

  AVAudioConverter *audioConverter = [[AVAudioConverter alloc] initFromFormat:kAudioEngineFormat
                                                                     toFormat:recordingFormat];

  // Load the audio record with the input buffer from the mock AVAudioEngine.
  AVAudioPCMBuffer *bufferInRecordingFormat =
      [MPPAudioRecord bufferFromInputBuffer:audioEngineBuffer
                        usingAudioConverter:audioConverter
                                      error:nil];

  return bufferInRecordingFormat;
}

#pragma mark Assertions for AudioData Creation

+ (MPPAudioData *)asssertCreateAudioDataWithChannelCount:(const NSUInteger)channelCount
                                              sampleRate:(const NSUInteger)sampleRate
                                             sampleCount:(const NSUInteger)sampleCount {
  MPPAudioDataFormat *audioDataFormat =
      [[MPPAudioDataFormat alloc] initWithChannelCount:channelCount sampleRate:sampleRate];

  MPPAudioData *audioData = [[MPPAudioData alloc] initWithFormat:audioDataFormat
                                                     sampleCount:sampleCount];
  AssertAudioDataIsValid(audioData, channelCount, sampleRate, sampleCount);
  return audioData;
}

#pragma mark Assertions for Load AudioData with FloatBuffer

// Convenience method to assert if loading audio data with input data of a certain length with
// offset 0 and all elements of input data succeeds and the resulting audio data buffer equals the
// expected data. Expected data must be of the same length as the audio data buffer.
+ (void)assertLoadSucceedsOnAudioData:(MPPAudioData *)audioData
                        withInputData:(const float *)inputData
                      inputDataLength:(NSUInteger)inputDataLength {
  [MPPAudioDataTests assertLoadSucceedsOnAudioData:audioData
                                     withInputData:inputData
                                   inputDataLength:inputDataLength
                                            offset:0
                                  lengthToBeLoaded:inputDataLength];
}

// Method to assert if loading audio record with input data of a certain length with the given
// offset and length of elements to be loaded succeeds and the resulting audio data buffer equals
// the expected data. Expected data must be of the same length as audio data buffer.
+ (void)assertLoadSucceedsOnAudioData:(MPPAudioData *)audioData
                        withInputData:(const float *)inputData
                      inputDataLength:(NSUInteger)inputDataLength
                               offset:(NSUInteger)offset
                     lengthToBeLoaded:(NSUInteger)lengthToBeLoaded {
  MPPFloatBuffer *previousStateOfAudioData = audioData.buffer;

  MPPFloatBuffer *inputBuffer = [[MPPFloatBuffer alloc] initWithData:inputData
                                                              length:inputDataLength];
  XCTAssertTrue([audioData loadBuffer:inputBuffer offset:offset length:lengthToBeLoaded error:nil]);

  MPPFloatBuffer *inputBufferFromOffset =
      [[MPPFloatBuffer alloc] initWithData:&(inputData[offset]) length:lengthToBeLoaded];
  [MPPAudioDataTests assertDataOfFloatBuffer:audioData.buffer
      containsInOrderSamplesFromPreviousStateOfFloatBuffer:previousStateOfAudioData
                                 andNewlyLoadedFloatBuffer:inputBufferFromOffset];
}

+ (void)assertFailureOfLoadAudioData:(MPPAudioData *)audioData
                            withData:(float *)inputData
                            ofLength:(NSUInteger)inputDataLength
                              offset:(NSUInteger)offset
                    lengthToBeLoaded:(NSUInteger)lengthToBeLoaded {
  NSError *error;

  MPPFloatBuffer *inputBuffer = [[MPPFloatBuffer alloc] initWithData:inputData
                                                              length:inputDataLength];
  XCTAssertFalse([audioData loadBuffer:inputBuffer
                                offset:offset
                                length:lengthToBeLoaded
                                 error:&error]);

  NSError *expectedError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey :
                   [NSString stringWithFormat:
                                 @"Index out of range. `offset` (%lu) + `length` (%lu) must be <= "
                                 @"`floatBuffer.length` (%lu)",

                                 offset, lengthToBeLoaded, inputDataLength]
             }];

  AssertEqualErrors(error, expectedError);
}

#pragma mark Assertions for Load AudioData with AudioRecord

+ (void)assertSuccessOfLoadAudioData:(MPPAudioData *)audioData
                     fromAudioRecord:(MPPAudioRecord *)audioRecord {
  MPPFloatBuffer *previousStateOfAudioData = audioData.buffer;

  XCTAssertTrue([audioData mockLoadAudioRecord:audioRecord error:nil]);

  MPPFloatBuffer *audioRecordBuffer = [audioRecord internalReadAtOffset:0
                                                             withLength:audioRecord.bufferLength
                                                                  error:nil];

  [MPPAudioDataTests assertDataOfFloatBuffer:audioData.buffer
      containsInOrderSamplesFromPreviousStateOfFloatBuffer:previousStateOfAudioData
                                 andNewlyLoadedFloatBuffer:audioRecordBuffer];
}

#pragma mark Helper Assertions

// Verifies if the current float buffer is created by shifting out old samples to make room for new
// samples from the new float buffer.
+ (void)assertDataOfFloatBuffer:(MPPFloatBuffer *)floatBuffer
    containsInOrderSamplesFromPreviousStateOfFloatBuffer:
        (MPPFloatBuffer *)previousStateOfFloatBuffer
                               andNewlyLoadedFloatBuffer:(MPPFloatBuffer *)newlyLoadedFloatBuffer {
  // The float buffer is compared in 2 chunks. If the float buffer is shorter than
  // `newlyLoadedFloatBuffer`, the len(first chunk) = 0 and len(second chunk) = full length of float
  // buffer i.e, the entire float buffer is compared with the most recent samples of length = float
  // buffer length in the `newlyLoadedFloatBuffer`. If the float buffer is longer than the
  // `newlyLoadedFloatBuffer`, the first chunk (oldest samples) must be equal to its previous state
  // offset from the len(newlyLoadedFloatBuffer) to verify that older elements are shifted out to
  // make room for new elements from the `newlyLoadedFloatBuffer`. len(second chunk) =
  // len(`newlyLoadedFloatBuffer`). All samples of `newlyLoadedFloatBuffer` must replace the samples
  // in the second chunk of the float buffer.
  const NSInteger secondChunkLength =
      std::min((NSInteger)floatBuffer.length, (NSInteger)newlyLoadedFloatBuffer.length);
  const NSInteger firstChunkLength = floatBuffer.length - secondChunkLength;

  for (int i = 0; i < firstChunkLength; i++) {
    XCTAssertEqualWithAccuracy(floatBuffer.data[i],
                               previousStateOfFloatBuffer.data[i + newlyLoadedFloatBuffer.length],
                               FLT_EPSILON);
  }

  // Starting indices for comparison of the second chunks in float buffer and
  // `newlyLoadedFloatBuffer` are calculated.
  const NSInteger startIndexForComparisonInFloatBuffer = firstChunkLength;
  const NSInteger startIndexForComparisonInNewlyLoadedBuffer =
      newlyLoadedFloatBuffer.length - secondChunkLength;

  for (int i = 0; i < secondChunkLength; i++) {
    XCTAssertEqualWithAccuracy(
        floatBuffer.data[startIndexForComparisonInFloatBuffer + i],
        newlyLoadedFloatBuffer.data[startIndexForComparisonInNewlyLoadedBuffer + i], FLT_EPSILON);
  }
}

@end

NS_ASSUME_NONNULL_END
