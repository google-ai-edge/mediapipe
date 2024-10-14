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

#import "mediapipe/tasks/ios/audio/audio_classifier/sources/MPPAudioClassifier.h"
#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/test/audio/core/utils/sources/AVAudioFile+TestUtils.h"
#import "mediapipe/tasks/ios/test/audio/core/utils/sources/AVAudioPCMBuffer+TestUtils.h"
#import "mediapipe/tasks/ios/test/audio/core/utils/sources/MPPAudioData+TestUtils.h"
#import "mediapipe/tasks/ios/test/utils/sources/MPPFileInfo.h"

static MPPFileInfo *const kYamnetModelFileInfo =
    [[MPPFileInfo alloc] initWithName:@"yamnet_audio_classifier_with_metadata" type:@"tflite"];
static MPPFileInfo *const kTwoHeadsModelFileInfo = [[MPPFileInfo alloc] initWithName:@"two_heads"
                                                                                type:@"tflite"];
static MPPFileInfo *const kSpeech16KHzMonoFileInfo =
    [[MPPFileInfo alloc] initWithName:@"speech_16000_hz_mono" type:@"wav"];
static MPPFileInfo *const kSpeech48KHzMonoFileInfo =
    [[MPPFileInfo alloc] initWithName:@"speech_48000_hz_mono" type:@"wav"];
static MPPFileInfo *const kTwoHeads16KHzMonoFileInfo =
    [[MPPFileInfo alloc] initWithName:@"two_heads_16000_hz_mono" type:@"wav"];
static MPPFileInfo *const kTwoHeads44KHzMonoFileInfo =
    [[MPPFileInfo alloc] initWithName:@"two_heads_44100_hz_mono" type:@"wav"];

static const NSInteger kYamnetCategoriesCount = 521;
static const NSInteger kYamnetClassificationResultsCount = 5;
static const NSInteger kYamnetSampleCount = 15600;
static const double kYamnetSampleRate = 16000.0;
static const NSInteger kMillisecondsPerSeconds = 1000;
static const NSInteger kYamnetIntervalSizeInMilliseconds =
    (NSInteger)((float)kYamnetSampleCount / kYamnetSampleRate * kMillisecondsPerSeconds);
static NSString *const kYamnetModelHeadName = @"scores";
static NSString *const kTwoHeadsModelYamnetHeadName = @"yamnet_classification";
static NSString *const kTwoHeadsModelBirdClassificationHeadName = @"bird_classification";

typedef NSDictionary<NSString *, NSNumber *> ClassificationHeadsCategoryCountInfo;

static ClassificationHeadsCategoryCountInfo *const kYamnetModelHeadsInfo =
    @{kYamnetModelHeadName : @(kYamnetCategoriesCount)};
static ClassificationHeadsCategoryCountInfo *const kTwoHeadModelHeadsInfo = @{
  kTwoHeadsModelYamnetHeadName : @(kYamnetCategoriesCount),
  kTwoHeadsModelBirdClassificationHeadName : @(5)
};

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";
static NSString *const kAudioStreamTestsDictClassifierKey = @"audioClassifier";
static NSString *const kAudioStreamTestsDictExpectationKey = @"expectation";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

#define AssertEqualCategories(category, expectedCategory, categoryIndex)                        \
  XCTAssertEqual(category.index, expectedCategory.index, @"index i = %d", categoryIndex);       \
  XCTAssertGreaterThan(category.score, expectedCategory.score, @"index i = %d", categoryIndex); \
  XCTAssertEqualObjects(category.categoryName, expectedCategory.categoryName, @"index i = %d",  \
                        categoryIndex);                                                         \
  XCTAssertEqualObjects(category.displayName, expectedCategory.displayName, @"index i = %d",    \
                        categoryIndex);

@interface MPPAudioClassifierTests : XCTestCase <MPPAudioClassifierStreamDelegate> {
  NSDictionary<NSString *, id> *_16kHZAudioStreamSucceedsTestDict;
  NSDictionary<NSString *, id> *_48kHZAudioStreamSucceedsTestDict;
  NSDictionary<NSString *, id> *_outOfOrderTimestampTestDict;
}
@end

@implementation MPPAudioClassifierTests

#pragma mark General Tests

- (void)testCreateAudioClassifierWithMissingModelPathFails {
  MPPFileInfo *fileInfo = [[MPPFileInfo alloc] initWithName:@"" type:@""];

  NSError *error = nil;
  MPPAudioClassifier *audioClassifier = [[MPPAudioClassifier alloc] initWithModelPath:fileInfo.path
                                                                                error:&error];
  XCTAssertNil(audioClassifier);

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

- (void)testCreateAudioClassifierAllowlistAndDenylistFails {
  MPPAudioClassifierOptions *options =
      [MPPAudioClassifierTests audioClassifierOptionsWithModelFileInfo:kTwoHeadsModelFileInfo];
  options.categoryAllowlist = @[ @"Speech" ];
  options.categoryDenylist = @[ @"Speech" ];

  [MPPAudioClassifierTests
      assertCreateAudioClassifierWithOptions:options
                      failsWithExpectedError:
                          [NSError
                              errorWithDomain:kExpectedErrorDomain
                                         code:MPPTasksErrorCodeInvalidArgumentError
                                     userInfo:@{
                                       NSLocalizedDescriptionKey :
                                           @"INVALID_ARGUMENT: `category_allowlist` and "
                                           @"`category_denylist` are mutually exclusive options."
                                     }]];
}

- (void)testClassifyWithYamnetAndModelPathSucceeds {
  MPPAudioClassifier *audioClassifier =
      [[MPPAudioClassifier alloc] initWithModelPath:kYamnetModelFileInfo.path error:nil];
  XCTAssertNotNil(audioClassifier);

  // Classify 48KHz speech file.
  [MPPAudioClassifierTests
          assertResultsOfClassifyAudioClipWithFileInfo:kSpeech48KHzMonoFileInfo
                                  usingAudioClassifier:audioClassifier
      approximatelyEqualsExpectedAudioClassifierResult:[MPPAudioClassifierTests
                                                           expectedPartialYamnetResult]
                    expectedClassificationResultsCount:kYamnetClassificationResultsCount
          expectedClassificationHeadsCategoryCountInfo:kYamnetModelHeadsInfo];

  // Classify 16KHz speech file.
  [MPPAudioClassifierTests
          assertResultsOfClassifyAudioClipWithFileInfo:kSpeech16KHzMonoFileInfo
                                  usingAudioClassifier:audioClassifier
      approximatelyEqualsExpectedAudioClassifierResult:[MPPAudioClassifierTests
                                                           expectedPartialYamnetResult]
                    expectedClassificationResultsCount:kYamnetClassificationResultsCount
          expectedClassificationHeadsCategoryCountInfo:kYamnetModelHeadsInfo];
}

- (void)testClassifyWithTwoHeadsAndOptionsSucceeds {
  MPPAudioClassifierOptions *options =
      [MPPAudioClassifierTests audioClassifierOptionsWithModelFileInfo:kTwoHeadsModelFileInfo];
  MPPAudioClassifier *audioClassifier =
      [MPPAudioClassifierTests audioClassifierWithOptions:options];

  // Classify 44KHz speech file.
  [MPPAudioClassifierTests
          assertResultsOfClassifyAudioClipWithFileInfo:kTwoHeads44KHzMonoFileInfo
                                  usingAudioClassifier:audioClassifier
      approximatelyEqualsExpectedAudioClassifierResult:[MPPAudioClassifierTests
                                                           expectedPartial44kHzTwoHeadsResult]
          expectedClassificationHeadsCategoryCountInfo:kTwoHeadModelHeadsInfo];

  // Classify 16KHz speech file.
  [MPPAudioClassifierTests
          assertResultsOfClassifyAudioClipWithFileInfo:kTwoHeads16KHzMonoFileInfo
                                  usingAudioClassifier:audioClassifier
      approximatelyEqualsExpectedAudioClassifierResult:[MPPAudioClassifierTests
                                                           expectedPartial16kHzTwoHeadsResult]
          expectedClassificationHeadsCategoryCountInfo:kTwoHeadModelHeadsInfo];
}

- (void)testClassifyWithMaxResultsSucceeds {
  const NSInteger maxResults = 1;

  MPPAudioClassifierOptions *options =
      [MPPAudioClassifierTests audioClassifierOptionsWithModelFileInfo:kYamnetModelFileInfo];
  options.maxResults = maxResults;

  MPPAudioClassifier *audioClassifier =
      [MPPAudioClassifierTests audioClassifierWithOptions:options];

  ClassificationHeadsCategoryCountInfo *const yamnetModelHeadsInfo =
      @{kYamnetModelHeadName : @(maxResults)};
  // Classify 16KHz speech file.
  [MPPAudioClassifierTests
          assertResultsOfClassifyAudioClipWithFileInfo:kSpeech16KHzMonoFileInfo
                                  usingAudioClassifier:audioClassifier
      approximatelyEqualsExpectedAudioClassifierResult:[MPPAudioClassifierTests
                                                           expectedPartialYamnetResult]
                    expectedClassificationResultsCount:kYamnetClassificationResultsCount
          expectedClassificationHeadsCategoryCountInfo:yamnetModelHeadsInfo];
}

- (void)testClassifyWithCategoryAllowlistSucceeds {
  MPPAudioClassifierOptions *options =
      [MPPAudioClassifierTests audioClassifierOptionsWithModelFileInfo:kYamnetModelFileInfo];
  options.categoryAllowlist = @[ @"Speech" ];

  MPPAudioClassifier *audioClassifier =
      [MPPAudioClassifierTests audioClassifierWithOptions:options];

  const NSInteger expectedCategoryCount = options.categoryAllowlist.count;
  ClassificationHeadsCategoryCountInfo *const yamnetModelHeadsInfo =
      @{kYamnetModelHeadName : @(expectedCategoryCount)};
  // Classify 16KHz speech file.
  [MPPAudioClassifierTests
          assertResultsOfClassifyAudioClipWithFileInfo:kSpeech16KHzMonoFileInfo
                                  usingAudioClassifier:audioClassifier
      approximatelyEqualsExpectedAudioClassifierResult:[MPPAudioClassifierTests
                                                           expectedPartialYamnetResult]
                    expectedClassificationResultsCount:kYamnetClassificationResultsCount
          expectedClassificationHeadsCategoryCountInfo:yamnetModelHeadsInfo];
}

- (void)testClassifyWithCategoryDenylistSucceeds {
  NSString *deniedCategory = @"Speech";
  MPPAudioClassifierOptions *options =
      [MPPAudioClassifierTests audioClassifierOptionsWithModelFileInfo:kYamnetModelFileInfo];
  options.categoryDenylist = @[ deniedCategory ];

  MPPAudioClassifier *audioClassifier =
      [MPPAudioClassifierTests audioClassifierWithOptions:options];

  // Classify 16KHz speech file.
  MPPAudioClassifierResult *result =
      [MPPAudioClassifierTests classifyAudioClipWithFileInfo:kSpeech16KHzMonoFileInfo
                                        usingAudioClassifier:audioClassifier];

  // Asserting that first category is not equal to `deniedCategory` in each `classificationResult`.
  XCTAssertEqual(result.classificationResults.count, kYamnetClassificationResultsCount);
  for (MPPClassificationResult *classificationResult in result.classificationResults) {
    XCTAssertEqual(classificationResult.classifications.count, 1);
    MPPClassifications *classifications = classificationResult.classifications[0];
    XCTAssertEqual(classifications.categories.count,
                   kYamnetCategoriesCount - options.categoryDenylist.count);
    XCTAssertFalse([classifications.categories[0].categoryName isEqualToString:deniedCategory]);
  }
}

- (void)testClassifyWithScoreThresholdSucceeds {
  MPPAudioClassifierOptions *options =
      [MPPAudioClassifierTests audioClassifierOptionsWithModelFileInfo:kYamnetModelFileInfo];
  options.scoreThreshold = 0.90f;

  MPPAudioClassifier *audioClassifier =
      [MPPAudioClassifierTests audioClassifierWithOptions:options];

  // Expecting only one category with a very high threshold.
  const NSInteger expectedCategoriesCount = 1;
  ClassificationHeadsCategoryCountInfo *const yamnetModelHeadsInfo =
      @{kYamnetModelHeadName : @(expectedCategoriesCount)};

  // Classify 16KHz speech file.
  [MPPAudioClassifierTests
          assertResultsOfClassifyAudioClipWithFileInfo:kSpeech16KHzMonoFileInfo
                                  usingAudioClassifier:audioClassifier
      approximatelyEqualsExpectedAudioClassifierResult:[MPPAudioClassifierTests
                                                           expectedPartialYamnetResult]
                    expectedClassificationResultsCount:kYamnetClassificationResultsCount
          expectedClassificationHeadsCategoryCountInfo:yamnetModelHeadsInfo];
}

- (void)testClassifyWithInsufficientDataSucceeds {
  MPPAudioClassifierOptions *options =
      [MPPAudioClassifierTests audioClassifierOptionsWithModelFileInfo:kYamnetModelFileInfo];
  MPPAudioClassifier *audioClassifier =
      [MPPAudioClassifierTests audioClassifierWithOptions:options];

  ClassificationHeadsCategoryCountInfo *const yamnetModelHeadsInfo =
      @{kYamnetModelHeadName : @(kYamnetCategoriesCount - options.categoryDenylist.count)};

  const NSInteger sampleCount = 14000;
  const double sampleRate = 16000;
  const NSInteger channelCount = 1;
  const NSInteger expectedClassificationResultsCount = 1;

  MPPAudioData *audioData = [[MPPAudioData alloc] initWithChannelCount:channelCount
                                                            sampleRate:sampleRate
                                                           sampleCount:sampleCount];

  MPPAudioClassifierResult *result = [audioClassifier classifyAudioClip:audioData error:nil];
  XCTAssertNotNil(result);

  [MPPAudioClassifierTests assertAudioClassifierResult:result
      approximatelyEqualToExpectedAudioClassifierResult:[MPPAudioClassifierTests
                                                            expectedYamnetInsufficientSilenceResult]
                     expectedClassificationResultsCount:expectedClassificationResultsCount
           expectedClassificationHeadsCategoryCountInfo:yamnetModelHeadsInfo];
}

- (void)testClassifyAfterCloseFailsInAudioClipsMode {
  MPPAudioClassifier *audioClassifier =
      [[MPPAudioClassifier alloc] initWithModelPath:kYamnetModelFileInfo.path error:nil];
  XCTAssertNotNil(audioClassifier);

  // Classify 48KHz speech file.
  [MPPAudioClassifierTests
          assertResultsOfClassifyAudioClipWithFileInfo:kSpeech48KHzMonoFileInfo
                                  usingAudioClassifier:audioClassifier
      approximatelyEqualsExpectedAudioClassifierResult:[MPPAudioClassifierTests
                                                           expectedPartialYamnetResult]
                    expectedClassificationResultsCount:kYamnetClassificationResultsCount
          expectedClassificationHeadsCategoryCountInfo:kYamnetModelHeadsInfo];

  NSError *closeError;
  XCTAssertTrue([audioClassifier closeWithError:&closeError]);
  XCTAssertNil(closeError);

  const NSInteger channelCount = 1;
  MPPAudioData *audioData = [[MPPAudioData alloc] initWithChannelCount:channelCount
                                                            sampleRate:kYamnetSampleRate
                                                           sampleCount:kYamnetSampleCount];

  NSError *classifyError;

  [audioClassifier classifyAudioClip:audioData error:&classifyError];

  NSError *expectedClassifyError = [NSError
      errorWithDomain:kExpectedErrorDomain
                 code:MPPTasksErrorCodeInvalidArgumentError
             userInfo:@{
               NSLocalizedDescriptionKey : [NSString
                   stringWithFormat:@"INVALID_ARGUMENT: Task runner is currently not running."]
             }];

  AssertEqualErrors(classifyError, expectedClassifyError);
}

- (void)testCreateAudioClassifierFailsWithDelegateInAudioClipsMode {
  MPPAudioClassifierOptions *options =
      [MPPAudioClassifierTests audioClassifierOptionsWithModelFileInfo:kYamnetModelFileInfo];
  options.audioClassifierStreamDelegate = self;

  [MPPAudioClassifierTests
      assertCreateAudioClassifierWithOptions:options
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

- (void)testClassifyFailsWithCallingWrongApiInAudioClipsMode {
  MPPAudioClassifierOptions *options =
      [MPPAudioClassifierTests audioClassifierOptionsWithModelFileInfo:kYamnetModelFileInfo];

  MPPAudioClassifier *audioClassifier =
      [MPPAudioClassifierTests audioClassifierWithOptions:options];

  MPPAudioData *audioClip = [[MPPAudioData alloc] initWithFileInfo:kSpeech16KHzMonoFileInfo];
  NSError *error;
  XCTAssertFalse([audioClassifier classifyAsyncAudioBlock:audioClip
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

- (void)testClassifyFailsWithCallingWrongApiInAudioStreamMode {
  MPPAudioClassifierOptions *options =
      [MPPAudioClassifierTests audioClassifierOptionsWithModelFileInfo:kYamnetModelFileInfo];
  options.runningMode = MPPAudioRunningModeAudioStream;
  options.audioClassifierStreamDelegate = self;

  MPPAudioClassifier *audioClassifier =
      [MPPAudioClassifierTests audioClassifierWithOptions:options];

  MPPAudioData *audioClip = [[MPPAudioData alloc] initWithFileInfo:kSpeech16KHzMonoFileInfo];

  NSError *error;
  XCTAssertFalse([audioClassifier classifyAudioClip:audioClip error:&error]);

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

- (void)testClassifyWithAudioStreamModeAndOutOfOrderTimestampsFails {
  MPPAudioClassifier *audioClassifier =
      [self audioClassifierInStreamModeWithModelFileInfo:kYamnetModelFileInfo];
  NSArray<MPPTimestampedAudioData *> *streamedAudioDataList =
      [MPPAudioClassifierTests streamedAudioDataListforYamnet];

  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"classifyWithOutOfOrderTimestampsAndLiveStream"];
  expectation.expectedFulfillmentCount = 1;

  _outOfOrderTimestampTestDict = @{
    kAudioStreamTestsDictClassifierKey : audioClassifier,
    kAudioStreamTestsDictExpectationKey : expectation
  };

  // Can safely access indices 1 and 0 `streamedAudioDataList` count is already asserted.
  XCTAssertTrue([audioClassifier
      classifyAsyncAudioBlock:streamedAudioDataList[1].audioData
      timestampInMilliseconds:streamedAudioDataList[1].timestampInMilliseconds
                        error:nil]);

  NSError *error;
  XCTAssertFalse([audioClassifier
      classifyAsyncAudioBlock:streamedAudioDataList[0].audioData
      timestampInMilliseconds:streamedAudioDataList[0].timestampInMilliseconds
                        error:&error]);

  NSError *expectedError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey :
                            @"INVALID_ARGUMENT: Input timestamp must be monotonically increasing."
                      }];
  AssertEqualErrors(error, expectedError);

  [audioClassifier closeWithError:nil];

  NSTimeInterval timeout = 1.0f;
  [self waitForExpectations:@[ expectation ] timeout:timeout];
}

- (void)testClassifyWithAudioStreamModeSucceeds {
  [self classifyUsingYamnetAsyncAudioFileWithInfo:kSpeech16KHzMonoFileInfo
                                             info:&_16kHZAudioStreamSucceedsTestDict];
  [self classifyUsingYamnetAsyncAudioFileWithInfo:kSpeech48KHzMonoFileInfo
                                             info:&_48kHZAudioStreamSucceedsTestDict];
}

#pragma mark Audio Record Tests

- (void)testCreateAudioRecordSucceeds {
  const NSUInteger channelCount = 1;
  const NSUInteger bufferLength = channelCount * kYamnetSampleCount;

  NSError *error;
  MPPAudioRecord *audioRecord =
      [MPPAudioClassifier createAudioRecordWithChannelCount:channelCount
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
      [MPPAudioClassifier createAudioRecordWithChannelCount:channelCount
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

#pragma mark MPPAudioClassifierStreamDelegate

- (void)audioClassifier:(MPPAudioClassifier *)audioClassifier
    didFinishClassificationWithResult:(MPPAudioClassifierResult *)result
              timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                error:(NSError *)error {
  // Can safely test for yamnet results before `audioClassifier` object tests since only yamnet with
  // 16khz and 48khz speech files are used for async tests.

  // Returns a `nil` `expectedResult` for the last timestamp to prevent the result from being
  // tested.
  MPPAudioClassifierResult *expectedResult = [MPPAudioClassifierTests
      expectedPartialYamnetResultWithTimestampInMilliseconds:timestampInMilliseconds
                                                isStreamMode:YES];

  // `expectedResult` will be `nil` for last timestamp since we are not testing for it.
  if (expectedResult) {
    [MPPAudioClassifierTests assertAudioClassifierResult:result
        approximatelyEqualToExpectedAudioClassifierResult:expectedResult
                       expectedClassificationResultsCount:1
             expectedClassificationHeadsCategoryCountInfo:kYamnetModelHeadsInfo];
  }

  if (audioClassifier == _outOfOrderTimestampTestDict[kAudioStreamTestsDictClassifierKey]) {
    [_outOfOrderTimestampTestDict[kAudioStreamTestsDictExpectationKey] fulfill];
  } else if (audioClassifier ==
             _16kHZAudioStreamSucceedsTestDict[kAudioStreamTestsDictClassifierKey]) {
    [_16kHZAudioStreamSucceedsTestDict[kAudioStreamTestsDictExpectationKey] fulfill];
  } else if (audioClassifier ==
             _48kHZAudioStreamSucceedsTestDict[kAudioStreamTestsDictClassifierKey]) {
    [_48kHZAudioStreamSucceedsTestDict[kAudioStreamTestsDictExpectationKey] fulfill];
  }
}

#pragma mark Audio Stream Mode Test Helpers

// info is strong here since address of global variables will be passed to this function. By default
// `NSDictionary **` will be `NSDictionary * __autoreleasing *.
- (void)classifyUsingYamnetAsyncAudioFileWithInfo:(MPPFileInfo *)audioFileInfo
                                             info:(NSDictionary<NSString *, id> *__strong *)info {
  MPPAudioClassifier *audioClassifier =
      [self audioClassifierInStreamModeWithModelFileInfo:kYamnetModelFileInfo];

  NSArray<MPPTimestampedAudioData *> *streamedAudioDataList =
      [MPPAudioClassifierTests streamedAudioDataListforYamnet];

  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:[NSString
                              stringWithFormat:@"classifyWithStreamMode_%@", audioFileInfo.name]];
  expectation.expectedFulfillmentCount = streamedAudioDataList.count;

  *info = @{
    kAudioStreamTestsDictClassifierKey : audioClassifier,
    kAudioStreamTestsDictExpectationKey : expectation
  };

  for (MPPTimestampedAudioData *timestampedAudioData in streamedAudioDataList) {
    XCTAssertTrue([audioClassifier
        classifyAsyncAudioBlock:timestampedAudioData.audioData
        timestampInMilliseconds:timestampedAudioData.timestampInMilliseconds
                          error:nil]);
  }

  [audioClassifier closeWithError:nil];

  NSTimeInterval timeout = 1.0f;
  [self waitForExpectations:@[ expectation ] timeout:timeout];
}

#pragma mark Audio Data Initializers

+ (NSArray<MPPTimestampedAudioData *> *)streamedAudioDataListforYamnet {
  NSArray<MPPTimestampedAudioData *> *streamedAudioDataList =
      [AVAudioFile streamedAudioBlocksFromAudioFileWithInfo:kSpeech16KHzMonoFileInfo
                                           modelSampleCount:kYamnetSampleCount
                                            modelSampleRate:kYamnetSampleRate];

  XCTAssertEqual(streamedAudioDataList.count, 5);

  return streamedAudioDataList;
}

#pragma mark Audio Classifier Initializers

- (MPPAudioClassifier *)audioClassifierInStreamModeWithModelFileInfo:(MPPFileInfo *)fileInfo {
  MPPAudioClassifierOptions *options =
      [MPPAudioClassifierTests audioClassifierOptionsWithModelFileInfo:fileInfo];
  options.runningMode = MPPAudioRunningModeAudioStream;
  options.audioClassifierStreamDelegate = self;

  MPPAudioClassifier *audioClassifier =
      [MPPAudioClassifierTests audioClassifierWithOptions:options];

  return audioClassifier;
}

+ (MPPAudioClassifierOptions *)audioClassifierOptionsWithModelFileInfo:
    (MPPFileInfo *)modelFileInfo {
  MPPAudioClassifierOptions *options = [[MPPAudioClassifierOptions alloc] init];
  options.baseOptions.modelAssetPath = modelFileInfo.path;

  return options;
}

+ (MPPAudioClassifier *)audioClassifierWithOptions:(MPPAudioClassifierOptions *)options {
  NSError *error;
  MPPAudioClassifier *audioClassifier = [[MPPAudioClassifier alloc] initWithOptions:options
                                                                              error:&error];
  XCTAssertNotNil(audioClassifier);
  XCTAssertNil(error);

  return audioClassifier;
}

+ (MPPAudioClassifier *)createAudioClassifierWithOptionsSucceeds:
    (MPPAudioClassifierOptions *)audioClassifierOptions {
  NSError *error;
  MPPAudioClassifier *audioClassifier =
      [[MPPAudioClassifier alloc] initWithOptions:audioClassifierOptions error:&error];
  XCTAssertNotNil(audioClassifier);
  XCTAssertNil(error);

  return audioClassifier;
}

+ (void)assertCreateAudioClassifierWithOptions:(MPPAudioClassifierOptions *)options
                        failsWithExpectedError:(NSError *)expectedError {
  NSError *error = nil;
  MPPAudioClassifier *audioClassifier = [[MPPAudioClassifier alloc] initWithOptions:options
                                                                              error:&error];

  XCTAssertNil(audioClassifier);
  AssertEqualErrors(error, expectedError);
}

#pragma mark Results

// Use this method when the `expectedAudioClassifierResult` defines an exhaustive list of
// `classificationResults` that are expected to be present in the predicted result. Eg: Expected Two
// heads results defines the full list of expected classification results whereas expected yamnet
// results do not.
+ (void)assertResultsOfClassifyAudioClipWithFileInfo:(MPPFileInfo *)fileInfo
                                usingAudioClassifier:(MPPAudioClassifier *)audioClassifier
    approximatelyEqualsExpectedAudioClassifierResult:
        (MPPAudioClassifierResult *)expectedAudioClassifierResult
        expectedClassificationHeadsCategoryCountInfo:
            (ClassificationHeadsCategoryCountInfo *)expectedCategoryCountInfo {
  [MPPAudioClassifierTests
          assertResultsOfClassifyAudioClipWithFileInfo:fileInfo
                                  usingAudioClassifier:audioClassifier
      approximatelyEqualsExpectedAudioClassifierResult:expectedAudioClassifierResult
                    expectedClassificationResultsCount:expectedAudioClassifierResult
                                                           .classificationResults.count
          expectedClassificationHeadsCategoryCountInfo:expectedCategoryCountInfo];
}

// Use this method when the `expectedAudioClassifierResult` defines only the first few
// `classificationResults` that are expected to be present in the predicted result. The no: of
// classification results in the predicted result must be compared with
// `expectedClassificationResultsCount` provided by the test.
+ (void)assertResultsOfClassifyAudioClipWithFileInfo:(MPPFileInfo *)fileInfo
                                usingAudioClassifier:(MPPAudioClassifier *)audioClassifier
    approximatelyEqualsExpectedAudioClassifierResult:
        (MPPAudioClassifierResult *)expectedAudioClassifierResult
                  expectedClassificationResultsCount:(NSInteger)expectedClassificationResultsCount
        expectedClassificationHeadsCategoryCountInfo:
            (ClassificationHeadsCategoryCountInfo *)expectedCategoryCountInfo {
  MPPAudioClassifierResult *result =
      [MPPAudioClassifierTests classifyAudioClipWithFileInfo:fileInfo
                                        usingAudioClassifier:audioClassifier];
  [MPPAudioClassifierTests assertAudioClassifierResult:result
      approximatelyEqualToExpectedAudioClassifierResult:expectedAudioClassifierResult
                     expectedClassificationResultsCount:expectedClassificationResultsCount
           expectedClassificationHeadsCategoryCountInfo:expectedCategoryCountInfo];
}

+ (MPPAudioClassifierResult *)classifyAudioClipWithFileInfo:(MPPFileInfo *)fileInfo
                                       usingAudioClassifier:(MPPAudioClassifier *)audioClassifier {
  MPPAudioData *audioData = [[MPPAudioData alloc] initWithFileInfo:fileInfo];
  MPPAudioClassifierResult *result = [audioClassifier classifyAudioClip:audioData error:nil];
  XCTAssertNotNil(result);

  return result;
}

+ (void)assertAudioClassifierResult:(MPPAudioClassifierResult *)result
    approximatelyEqualToExpectedAudioClassifierResult:(MPPAudioClassifierResult *)expectedResult
                   expectedClassificationResultsCount:(NSInteger)expectedClassificationResultsCount
         expectedClassificationHeadsCategoryCountInfo:
             (ClassificationHeadsCategoryCountInfo *)expectedCategoryCountInfo {
  XCTAssertEqual(result.classificationResults.count, expectedClassificationResultsCount);

  for (int index = 0; index < expectedResult.classificationResults.count; ++index) {
    MPPClassificationResult *classificationResult = result.classificationResults[index];
    MPPClassificationResult *expectedClassificationResult =
        expectedResult.classificationResults[index];
    XCTAssertEqual(classificationResult.timestampInMilliseconds,
                   expectedClassificationResult.timestampInMilliseconds);
    XCTAssertEqual(classificationResult.classifications.count,
                   expectedClassificationResult.classifications.count);

    for (int classificationIndex = 0;
         classificationIndex < expectedClassificationResult.classifications.count;
         ++classificationIndex) {
      MPPClassifications *expectedClassifications =
          expectedClassificationResult.classifications[classificationIndex];
      [MPPAudioClassifierTests assertClassificationHead:classificationResult
                                                            .classifications[classificationIndex]
            approximatelyEqualToExpectedClassifications:expectedClassifications
                                  expectedCategoryCount:expectedCategoryCountInfo
                                                            [expectedClassifications.headName]
                                                                .intValue];
    }
  }
}

+ (void)assertClassificationHead:(MPPClassifications *)classifications
    approximatelyEqualToExpectedClassifications:(MPPClassifications *)expectedClassifications
                          expectedCategoryCount:(NSInteger)expectedCategoryCount {
  XCTAssertEqual(classifications.headIndex, expectedClassifications.headIndex);
  XCTAssertEqualObjects(classifications.headName, expectedClassifications.headName);
  XCTAssertEqual(classifications.categories.count, expectedCategoryCount);

  // If `expectedCategoryCount` = count of all possible categories, only compare the with the number
  // of categories in `expectedClassifications` since `expectedClassifications` defined in the tests
  // does not contain an exhaustive list of predictions for all possible categories.
  // If `expectedCategoryCount` = maxResults, `expectedClassifications` may have more no: of
  // categories than the predicted classifications. Hence compare the first minimum no: of
  // categories amongst `expectedCategoryCount` and `expectedClassifications.categories.count`.
  NSInteger categoriesToVerifyCount =
      MIN(expectedCategoryCount, expectedClassifications.categories.count);
  for (int categoryIndex = 0; categoryIndex < categoriesToVerifyCount; ++categoryIndex) {
    AssertEqualCategories(classifications.categories[categoryIndex],
                          expectedClassifications.categories[categoryIndex], categoryIndex);
  }
}

+ (MPPAudioClassifierResult *)expectedPartialYamnetResult {
  return [MPPAudioClassifierTests expectedPartialYamnetResultWithTimestampInMilliseconds:0
                                                                            isStreamMode:NO];
}

// Returns only one top category for each classification head.
// Last classification result (timestamped result) is omitted because it varies between test
// runs due to the low confidence score. Ensure that the subset of classification results in the
// predicted audio classifier result is compared with the expected result returned from this method.
// If `isStream` mode is set, returned result will only have the `classificationResult` for the
// given `timestampInMilliseconds`.
+ (MPPAudioClassifierResult *)
    expectedPartialYamnetResultWithTimestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                              isStreamMode:(BOOL)isStreamMode {
  const NSInteger maxTimestampToCompare = 2925;
  const NSInteger minTimestampToCompare = 0;

  // Last timestamp and any other illegal values of timestamp are not allowed to pass through.
  if (timestampInMilliseconds > maxTimestampToCompare ||
      timestampInMilliseconds < minTimestampToCompare ||
      timestampInMilliseconds % kYamnetIntervalSizeInMilliseconds != 0) {
    return nil;
  }

  // Only one of the classification results corresponding to the given  timestamp is to be returned
  // as the expected result for stream mode. Calculate index of the `classificationResult` to be
  // returned based on the timestamp and the input size of the Yamnet model in milliseconds.
  NSInteger index = timestampInMilliseconds / kYamnetIntervalSizeInMilliseconds;

  NSArray<MPPClassificationResult *> *classificationResults = @[
    [[MPPClassificationResult alloc] initWithClassifications:@[
      [[MPPClassifications alloc] initWithHeadIndex:0
                                           headName:kYamnetModelHeadName
                                         categories:@[ [[MPPCategory alloc] initWithIndex:0
                                                                                    score:0.90f
                                                                             categoryName:@"Speech"
                                                                              displayName:nil] ]]
    ]
                                     timestampInMilliseconds:0],
    [[MPPClassificationResult alloc] initWithClassifications:@[
      [[MPPClassifications alloc] initWithHeadIndex:0
                                           headName:kYamnetModelHeadName
                                         categories:@[ [[MPPCategory alloc] initWithIndex:0
                                                                                    score:0.90f
                                                                             categoryName:@"Speech"
                                                                              displayName:nil] ]]
    ]
                                     timestampInMilliseconds:975],
    [[MPPClassificationResult alloc] initWithClassifications:@[
      [[MPPClassifications alloc] initWithHeadIndex:0
                                           headName:kYamnetModelHeadName
                                         categories:@[ [[MPPCategory alloc] initWithIndex:0
                                                                                    score:0.90f
                                                                             categoryName:@"Speech"
                                                                              displayName:nil] ]]
    ]
                                     timestampInMilliseconds:1950],
    [[MPPClassificationResult alloc] initWithClassifications:@[
      [[MPPClassifications alloc] initWithHeadIndex:0
                                           headName:kYamnetModelHeadName
                                         categories:@[ [[MPPCategory alloc] initWithIndex:0
                                                                                    score:0.90f
                                                                             categoryName:@"Speech"
                                                                              displayName:nil] ]]
    ]
                                     timestampInMilliseconds:2925],

  ];

  // In stream mode, only one classification result corresponding to the requested timestamp is
  // returned. In clips mode, the full array of classification results are returned.
  return [[MPPAudioClassifierResult alloc]
      initWithClassificationResults:isStreamMode ? @[ classificationResults[index] ]
                                                 : classificationResults
            timestampInMilliseconds:timestampInMilliseconds];
}

+ (MPPAudioClassifierResult *)expectedPartial44kHzTwoHeadsResult {
  return [MPPAudioClassifierTests expectedPartial44kHzTwoHeadsResultWithTimestampInMilliseconds:0];
}

+ (MPPAudioClassifierResult *)expectedPartial16kHzTwoHeadsResult {
  return [MPPAudioClassifierTests expectedPartial16kHzTwoHeadsResultWithTimestampInMilliseconds:0];
}

+ (MPPAudioClassifierResult *)expectedPartial44kHzTwoHeadsResultWithTimestampInMilliseconds:
    (NSInteger)timestampInMilliseconds {
  NSArray<MPPClassificationResult *> *classificationResults = @[
    [[MPPClassificationResult alloc] initWithClassifications:@[
      [[MPPClassifications alloc]
          initWithHeadIndex:0
                   headName:kTwoHeadsModelYamnetHeadName
                 categories:@[ [[MPPCategory alloc] initWithIndex:508
                                                            score:0.50f
                                                     categoryName:@"Environmental noise"
                                                      displayName:nil] ]],
      [[MPPClassifications alloc]
          initWithHeadIndex:1
                   headName:kTwoHeadsModelBirdClassificationHeadName
                 categories:@[ [[MPPCategory alloc] initWithIndex:4
                                                            score:0.93f
                                                     categoryName:@"Chestnut-crowned Antpitta"
                                                      displayName:nil] ]],
    ]
                                     timestampInMilliseconds:0],
    [[MPPClassificationResult alloc] initWithClassifications:@[
      [[MPPClassifications alloc] initWithHeadIndex:0
                                           headName:kTwoHeadsModelYamnetHeadName
                                         categories:@[ [[MPPCategory alloc] initWithIndex:494
                                                                                    score:0.90f
                                                                             categoryName:@"Silence"
                                                                              displayName:nil] ]],
      [[MPPClassifications alloc]
          initWithHeadIndex:1
                   headName:kTwoHeadsModelBirdClassificationHeadName
                 categories:@[ [[MPPCategory alloc] initWithIndex:1
                                                            score:0.99f
                                                     categoryName:@"White-breasted Wood-Wren"
                                                      displayName:nil] ]]
    ]
                                     timestampInMilliseconds:975],

  ];

  return [[MPPAudioClassifierResult alloc] initWithClassificationResults:classificationResults
                                                 timestampInMilliseconds:timestampInMilliseconds];
}

+ (MPPAudioClassifierResult *)expectedPartial16kHzTwoHeadsResultWithTimestampInMilliseconds:
    (NSInteger)timestampInMilliseconds {
  NSArray<MPPClassificationResult *> *classificationResults = @[
    [[MPPClassificationResult alloc] initWithClassifications:@[
      [[MPPClassifications alloc]
          initWithHeadIndex:0
                   headName:kTwoHeadsModelYamnetHeadName
                 categories:@[ [[MPPCategory alloc] initWithIndex:508
                                                            score:0.50f
                                                     categoryName:@"Environmental noise"
                                                      displayName:nil] ]],
      [[MPPClassifications alloc]
          initWithHeadIndex:1
                   headName:kTwoHeadsModelBirdClassificationHeadName
                 categories:@[ [[MPPCategory alloc] initWithIndex:4
                                                            score:0.93f
                                                     categoryName:@"Chestnut-crowned Antpitta"
                                                      displayName:nil] ]],
    ]
                                     timestampInMilliseconds:0],
  ];

  return [[MPPAudioClassifierResult alloc] initWithClassificationResults:classificationResults
                                                 timestampInMilliseconds:timestampInMilliseconds];
}

+ (MPPAudioClassifierResult *)expectedYamnetInsufficientSilenceResult {
  NSArray<MPPClassificationResult *> *classificationResults = @[
    [[MPPClassificationResult alloc] initWithClassifications:@[
      [[MPPClassifications alloc] initWithHeadIndex:0
                                           headName:kYamnetModelHeadName
                                         categories:@[ [[MPPCategory alloc] initWithIndex:494
                                                                                    score:0.8f
                                                                             categoryName:@"Silence"
                                                                              displayName:nil] ]],
    ]
                                     timestampInMilliseconds:0],
  ];

  return [[MPPAudioClassifierResult alloc] initWithClassificationResults:classificationResults
                                                 timestampInMilliseconds:0];
}

@end
