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

#import <XCTest/XCTest.h>

#import "mediapipe/tasks/ios/audio/audio_Embedder/sources/MPPAudioEmbedder.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/test/audio/core/utils/sources/AVAudioFile+TestUtils.h"
#import "mediapipe/tasks/ios/test/audio/core/utils/sources/AVAudioPCMBuffer+TestUtils.h"
#import "mediapipe/tasks/ios/test/audio/core/utils/sources/MPPAudioData+TestUtils.h"
#import "mediapipe/tasks/ios/test/utils/sources/MPPFileInfo.h"

static MPPFileInfo *const kYamnetModelFileInfo =
    [[MPPFileInfo alloc] initWithName:@"yamnet_embedding_metadata" type:@"tflite"];
static MPPFileInfo *const kSpeech16KHzMonoFileInfo =
    [[MPPFileInfo alloc] initWithName:@"speech_16000_hz_mono" type:@"wav"];
static MPPFileInfo *const kSpeech48KHzMonoFileInfo =
    [[MPPFileInfo alloc] initWithName:@"speech_48000_hz_mono" type:@"wav"];
static MPPFileInfo *const kTwoHeads16KHzMonoFileInfo =
    [[MPPFileInfo alloc] initWithName:@"two_heads_16000_hz_mono" type:@"wav"];

static const NSInteger kYamnetSampleCount = 15600;
static const double kYamnetSampleRate = 16000.0;
static const NSInteger kExpectedEmbeddingLength = 1024;
static const NSInteger kExpectedEmbeddingResultsCountForSpeechFiles = 5;

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";
static NSString *const kAudioStreamTestsDictEmbedderKey = @"audioEmbedder";
static NSString *const kAudioStreamTestsDictExpectationKey = @"expectation";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

#define AssertEmbeddingResultHasOneEmbedding(embeddingResult) \
  XCTAssertNotNil(embeddingResult);                           \
  \                                                         
  XCTAssertEqual(embeddingResult.embeddings.count, 1);

#define AssertEmbeddingHasCorrectTypeAndDimension(embedding, quantize, expectedLength) \
  if (quantize) {                                                                      \
    XCTAssertNil(embedding.floatEmbedding);                                            \
    XCTAssertNotNil(embedding.quantizedEmbedding);                                     \
    XCTAssertEqual(embedding.quantizedEmbedding.count, expectedLength);                \
  } else {                                                                             \
    XCTAssertNotNil(embedding.floatEmbedding);                                         \
    XCTAssertNil(embedding.quantizedEmbedding);                                        \
    XCTAssertEqual(embedding.floatEmbedding.count, expectedLength);                    \
  }

@interface MPPAudioEmbedderTests : XCTestCase <MPPAudioEmbedderStreamDelegate>
@end

@implementation MPPAudioEmbedderTests

#pragma mark General Tests

- (void)testCreateAudioEmbedderWithMissingModelPathFails {
  MPPFileInfo *fileInfo = [[MPPFileInfo alloc] initWithName:@"" type:@""];

  NSError *error = nil;
  MPPAudioEmbedder *audioEmbedder = [[MPPAudioEmbedder alloc] initWithModelPath:fileInfo.path
                                                                          error:&error];
  XCTAssertNil(audioEmbedder);

  NSError *expectedError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey :
                   @"INVALID_ARGUMENT: ExternalFile must specify at least one of 'file_content', "
                   @"'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."
             }];
  AssertEqualErrors(error, expectedError);
}

- (void)testEmbedWithModelPathAndDifferentAudioFilesSucceeds {
  MPPAudioEmbedder *audioEmbedder =
      [[MPPAudioEmbedder alloc] initWithModelPath:kYamnetModelFileInfo.path error:nil];
  XCTAssertNotNil(audioEmbedder);

  [MPPAudioEmbedderTests
      assertResultsOfNonQuantizedEmbedAudioClipWithFileInfo:kSpeech16KHzMonoFileInfo
                                         usingAudioEmbedder:audioEmbedder
                              expectedEmbeddingResultsCount:
                                  kExpectedEmbeddingResultsCountForSpeechFiles];
  [MPPAudioEmbedderTests
      assertResultsOfNonQuantizedEmbedAudioClipWithFileInfo:kSpeech48KHzMonoFileInfo
                                         usingAudioEmbedder:audioEmbedder
                              expectedEmbeddingResultsCount:
                                  kExpectedEmbeddingResultsCountForSpeechFiles];

  const NSInteger expectedEmbeddingResultCount = 1;
  [MPPAudioEmbedderTests
      assertResultsOfNonQuantizedEmbedAudioClipWithFileInfo:kTwoHeads16KHzMonoFileInfo
                                         usingAudioEmbedder:audioEmbedder
                              expectedEmbeddingResultsCount:expectedEmbeddingResultCount];
}

- (void)testEmbedWithOptionsSucceeds {
  MPPAudioEmbedderOptions *options =
      [MPPAudioEmbedderTests audioEmbedderOptionsWithModelFileInfo:kYamnetModelFileInfo];

  MPPAudioEmbedder *audioEmbedder = [MPPAudioEmbedderTests audioEmbedderWithOptions:options];

  [MPPAudioEmbedderTests
      assertResultsOfNonQuantizedEmbedAudioClipWithFileInfo:kSpeech16KHzMonoFileInfo
                                         usingAudioEmbedder:audioEmbedder
                              expectedEmbeddingResultsCount:
                                  kExpectedEmbeddingResultsCountForSpeechFiles];
  [MPPAudioEmbedderTests
      assertResultsOfNonQuantizedEmbedAudioClipWithFileInfo:kSpeech48KHzMonoFileInfo
                                         usingAudioEmbedder:audioEmbedder
                              expectedEmbeddingResultsCount:
                                  kExpectedEmbeddingResultsCountForSpeechFiles];
}

- (void)testEmbedWithQuantizationSucceeds {
  MPPAudioEmbedderOptions *options =
      [MPPAudioEmbedderTests audioEmbedderOptionsWithModelFileInfo:kYamnetModelFileInfo];
  options.quantize = YES;

  MPPAudioEmbedder *audioEmbedder = [MPPAudioEmbedderTests audioEmbedderWithOptions:options];

  const NSInteger expectedEmbeddingResultsCount = 5;
  [MPPAudioEmbedderTests assertResultsOfEmbedAudioClipWithFileInfo:kSpeech16KHzMonoFileInfo
                                                usingAudioEmbedder:audioEmbedder
                                                       isQuantized:options.quantize
                                     expectedEmbeddingResultsCount:expectedEmbeddingResultsCount];
  [MPPAudioEmbedderTests assertResultsOfEmbedAudioClipWithFileInfo:kSpeech48KHzMonoFileInfo
                                                usingAudioEmbedder:audioEmbedder
                                                       isQuantized:options.quantize
                                     expectedEmbeddingResultsCount:expectedEmbeddingResultsCount];
}

- (void)testEmbedWithL2NormalizationSucceeds {
  MPPAudioEmbedderOptions *options =
      [MPPAudioEmbedderTests audioEmbedderOptionsWithModelFileInfo:kYamnetModelFileInfo];
  options.l2Normalize = YES;

  MPPAudioEmbedder *audioEmbedder = [MPPAudioEmbedderTests audioEmbedderWithOptions:options];

  [MPPAudioEmbedderTests
      assertResultsOfNonQuantizedEmbedAudioClipWithFileInfo:kSpeech16KHzMonoFileInfo
                                         usingAudioEmbedder:audioEmbedder
                              expectedEmbeddingResultsCount:
                                  kExpectedEmbeddingResultsCountForSpeechFiles];
  [MPPAudioEmbedderTests
      assertResultsOfNonQuantizedEmbedAudioClipWithFileInfo:kSpeech48KHzMonoFileInfo
                                         usingAudioEmbedder:audioEmbedder
                              expectedEmbeddingResultsCount:
                                  kExpectedEmbeddingResultsCountForSpeechFiles];
}

- (void)testEmbedWithSilenceSucceeds {
  MPPAudioEmbedderOptions *options =
      [MPPAudioEmbedderTests audioEmbedderOptionsWithModelFileInfo:kYamnetModelFileInfo];
  MPPAudioEmbedder *audioEmbedder = [MPPAudioEmbedderTests audioEmbedderWithOptions:options];

  const NSInteger channelCount = 1;

  MPPAudioData *audioData = [[MPPAudioData alloc] initWithChannelCount:channelCount
                                                            sampleRate:kYamnetSampleRate
                                                           sampleCount:kYamnetSampleCount];

  MPPAudioEmbedderResult *result = [audioEmbedder embedAudioClip:audioData error:nil];
  XCTAssertNotNil(result);

  const NSInteger expectedEmbedderResultsCount = 1;
  [MPPAudioEmbedderTests assertAudioEmbedderResult:result
                                       isQuantized:options.quantize
                     expectedEmbeddingResultsCount:expectedEmbedderResultsCount
                           expectedEmbeddingLength:kExpectedEmbeddingLength];
}

- (void)testEmbedAfterCloseFailsInAudioClipsMode {
  MPPAudioEmbedderOptions *options =
      [MPPAudioEmbedderTests audioEmbedderOptionsWithModelFileInfo:kYamnetModelFileInfo];
  MPPAudioEmbedder *audioEmbedder = [MPPAudioEmbedderTests audioEmbedderWithOptions:options];

  // Classify 16KHz speech file.
  [MPPAudioEmbedderTests
      assertResultsOfEmbedAudioClipWithFileInfo:kSpeech16KHzMonoFileInfo
                             usingAudioEmbedder:audioEmbedder
                                    isQuantized:options.quantize
                  expectedEmbeddingResultsCount:kExpectedEmbeddingResultsCountForSpeechFiles];

  NSError *closeError;
  XCTAssertTrue([audioEmbedder closeWithError:&closeError]);
  XCTAssertNil(closeError);

  const NSInteger channelCount = 1;
  MPPAudioData *audioData = [[MPPAudioData alloc] initWithChannelCount:channelCount
                                                            sampleRate:kYamnetSampleRate
                                                           sampleCount:kYamnetSampleCount];

  NSError *embedError;

  [audioEmbedder embedAudioClip:audioData error:&embedError];

  NSError *expectedEmbedError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey : [NSString
                   stringWithFormat:@"INVALID_ARGUMENT: Task runner is currently not running."]
             }];

  AssertEqualErrors(embedError, expectedEmbedError);
}

#pragma mark Running mode tests

- (void)testCreateAudioEmbedderFailsWithDelegateInAudioClipsMode {
  MPPAudioEmbedderOptions *options =
      [MPPAudioEmbedderTests audioEmbedderOptionsWithModelFileInfo:kYamnetModelFileInfo];
  options.audioEmbedderStreamDelegate = self;

  [MPPAudioEmbedderTests
      assertCreateAudioEmbedderWithOptions:options
                    failsWithExpectedError:
                        [NSError
                            errorWithDomain:kExpectedErrorDomain
                                       code:MPPTasksErrorCodeInvalidArgumentError
                                   userInfo:@{
                                     NSLocalizedDescriptionKey : [NSString
                                         stringWithFormat:@"The audio task is in audio clips "
                                                          @"mode. The delegate must not be set "
                                                          @"in the task's options."]
                                   }]];
}

- (void)testEmbedFailsWithCallingWrongApiInAudioClipsMode {
  MPPAudioEmbedderOptions *options =
      [MPPAudioEmbedderTests audioEmbedderOptionsWithModelFileInfo:kYamnetModelFileInfo];

  MPPAudioEmbedder *audioEmbedder = [MPPAudioEmbedderTests audioEmbedderWithOptions:options];

  MPPAudioData *audioClip = [[MPPAudioData alloc] initWithFileInfo:kSpeech16KHzMonoFileInfo];
  NSError *error;
  XCTAssertFalse([audioEmbedder embedAsyncAudioBlock:audioClip
                             timestampInMilliseconds:0
                                               error:&error]);

  NSError *expectedError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey :
                   [NSString stringWithFormat:@"The audio task is not initialized with "
                                              @"audio stream mode. Current Running Mode: %@",
                                              MPPAudioRunningModeDisplayName(options.runningMode)]
             }];
  AssertEqualErrors(error, expectedError);
}

- (void)testEmbedFailsWithCallingWrongApiInAudioStreamMode {
  MPPAudioEmbedderOptions *options =
      [MPPAudioEmbedderTests audioEmbedderOptionsWithModelFileInfo:kYamnetModelFileInfo];
  options.runningMode = MPPAudioRunningModeAudioStream;
  options.audioEmbedderStreamDelegate = self;

  MPPAudioEmbedder *audioEmbedder = [MPPAudioEmbedderTests audioEmbedderWithOptions:options];

  MPPAudioData *audioClip = [[MPPAudioData alloc] initWithFileInfo:kSpeech16KHzMonoFileInfo];

  NSError *error;
  XCTAssertFalse([audioEmbedder embedAudioClip:audioClip error:&error]);

  NSError *expectedError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey :
                   [NSString stringWithFormat:@"The audio task is not initialized with "
                                              @"audio clips. Current Running Mode: %@",
                                              MPPAudioRunningModeDisplayName(options.runningMode)]
             }];
  AssertEqualErrors(error, expectedError);
}

#pragma mark Audio Record Tests

- (void)testCreateAudioRecordSucceeds {
  const NSUInteger channelCount = 1;
  const NSUInteger bufferLength = channelCount * kYamnetSampleCount;

  NSError *error;
  MPPAudioRecord *audioRecord =
      [MPPAudioEmbedder createAudioRecordWithChannelCount:channelCount
                                               sampleRate:kYamnetSampleRate
                                             bufferLength:kYamnetSampleCount * channelCount
                                                    error:&error];

  XCTAssertNotNil(audioRecord);
  XCTAssertNil(error);
  XCTAssertEqual(audioRecord.audioDataFormat.channelCount, channelCount);
  XCTAssertEqual(audioRecord.audioDataFormat.sampleRate, kYamnetSampleRate);
  XCTAssertEqual(audioRecord.bufferLength, bufferLength);
}

// Test for error propogation from audio record creation.
- (void)testCreateAudioRecordWithInvalidChannelCountFails {
  const NSUInteger channelCount = 3;

  NSError *error;
  MPPAudioRecord *audioRecord =
      [MPPAudioEmbedder createAudioRecordWithChannelCount:channelCount
                                               sampleRate:kYamnetSampleRate
                                             bufferLength:kYamnetSampleCount * channelCount
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

#pragma mark MPPAudioEmbedderStreamDelegate

- (void)audioEmbedder:(MPPAudioEmbedder *)audioEmbedder
    didFinishEmbeddingWithResult:(MPPAudioEmbedderResult *)result
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(NSError *)error {
  // TODO: Add assertion for the result when stream mode inference tests are added.
}

#pragma mark Audio Embedder Initializers

+ (MPPAudioEmbedderOptions *)audioEmbedderOptionsWithModelFileInfo:(MPPFileInfo *)modelFileInfo {
  MPPAudioEmbedderOptions *options = [[MPPAudioEmbedderOptions alloc] init];
  options.baseOptions.modelAssetPath = modelFileInfo.path;

  return options;
}

+ (MPPAudioEmbedder *)audioEmbedderWithOptions:(MPPAudioEmbedderOptions *)options {
  NSError *error;
  MPPAudioEmbedder *audioEmbedder = [[MPPAudioEmbedder alloc] initWithOptions:options error:&error];
  XCTAssertNotNil(audioEmbedder);
  XCTAssertNil(error);

  return audioEmbedder;
}

+ (MPPAudioEmbedder *)createAudioEmbedderWithOptionsSucceeds:
    (MPPAudioEmbedderOptions *)audioEmbedderOptions {
  NSError *error;
  MPPAudioEmbedder *audioEmbedder = [[MPPAudioEmbedder alloc] initWithOptions:audioEmbedderOptions
                                                                        error:&error];
  XCTAssertNotNil(audioEmbedder);
  XCTAssertNil(error);

  return audioEmbedder;
}

+ (void)assertCreateAudioEmbedderWithOptions:(MPPAudioEmbedderOptions *)options
                      failsWithExpectedError:(NSError *)expectedError {
  NSError *error = nil;
  MPPAudioEmbedder *audioEmbedder = [[MPPAudioEmbedder alloc] initWithOptions:options error:&error];

  XCTAssertNil(audioEmbedder);
  AssertEqualErrors(error, expectedError);
}

#pragma mark Results

+ (void)assertResultsOfNonQuantizedEmbedAudioClipWithFileInfo:(MPPFileInfo *)fileInfo
                                           usingAudioEmbedder:(MPPAudioEmbedder *)audioEmbedder
                                expectedEmbeddingResultsCount:
                                    (NSInteger)expectedEmbeddingResultsCount {
  [MPPAudioEmbedderTests assertResultsOfEmbedAudioClipWithFileInfo:fileInfo
                                                usingAudioEmbedder:audioEmbedder
                                                       isQuantized:NO
                                     expectedEmbeddingResultsCount:expectedEmbeddingResultsCount];
}

+ (void)assertResultsOfEmbedAudioClipWithFileInfo:(MPPFileInfo *)fileInfo
                               usingAudioEmbedder:(MPPAudioEmbedder *)audioEmbedder
                                      isQuantized:(BOOL)isQuantized
                    expectedEmbeddingResultsCount:(NSInteger)expectedEmbeddingResultsCount {
  MPPAudioEmbedderResult *result = [MPPAudioEmbedderTests embedAudioClipWithFileInfo:fileInfo
                                                                  usingAudioEmbedder:audioEmbedder];

  [MPPAudioEmbedderTests assertAudioEmbedderResult:result
                                       isQuantized:isQuantized
                     expectedEmbeddingResultsCount:expectedEmbeddingResultsCount
                           expectedEmbeddingLength:kExpectedEmbeddingLength];
}

+ (void)assertAudioEmbedderResult:(MPPAudioEmbedderResult *)result
                      isQuantized:(BOOL)isQuantized
    expectedEmbeddingResultsCount:(NSInteger)expectedEmbeddingResultsCount
          expectedEmbeddingLength:(NSInteger)expectedEmbeddingLength {
  XCTAssertEqual(result.embeddingResults.count, expectedEmbeddingResultsCount);
  for (MPPEmbeddingResult *embeddingResult in result.embeddingResults) {
    AssertEmbeddingResultHasOneEmbedding(embeddingResult);
    AssertEmbeddingHasCorrectTypeAndDimension(embeddingResult.embeddings[0], isQuantized,
                                              expectedEmbeddingLength);
  }
}

+ (MPPAudioEmbedderResult *)embedAudioClipWithFileInfo:(MPPFileInfo *)fileInfo
                                    usingAudioEmbedder:(MPPAudioEmbedder *)audioEmbedder {
  MPPAudioData *audioData = [[MPPAudioData alloc] initWithFileInfo:fileInfo];
  MPPAudioEmbedderResult *result = [audioEmbedder embedAudioClip:audioData error:nil];
  XCTAssertNotNil(result);

  return result;
}

@end
