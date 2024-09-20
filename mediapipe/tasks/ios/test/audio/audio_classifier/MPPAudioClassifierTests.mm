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
#import "mediapipe/tasks/ios/test/audio/core/utils/sources/AVAudioPCMBuffer+TestUtils.h"
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

@interface MPPAudioClassifierTests : XCTestCase <MPPAudioClassifierStreamDelegate>
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

  const NSInteger expectedCategoryCount = 1;
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

  ClassificationHeadsCategoryCountInfo *const yamnetModelHeadsInfo =
      @{kYamnetModelHeadName : @(kYamnetCategoriesCount - options.categoryDenylist.count)};

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

- (void)testClassifyWithInsufficientDataSucceeds {
  MPPAudioClassifierOptions *options =
      [MPPAudioClassifierTests audioClassifierOptionsWithModelFileInfo:kYamnetModelFileInfo];
  MPPAudioClassifier *audioClassifier =
      [MPPAudioClassifierTests audioClassifierWithOptions:options];

  ClassificationHeadsCategoryCountInfo *const yamnetModelHeadsInfo =
      @{kYamnetModelHeadName : @(kYamnetCategoriesCount - options.categoryDenylist.count)};

  MPPAudioDataFormat *format = [[MPPAudioDataFormat alloc] initWithChannelCount:1 sampleRate:16000];
  MPPAudioData *audioData = [[MPPAudioData alloc] initWithFormat:format sampleCount:14000];

  MPPAudioClassifierResult *result = [audioClassifier classifyAudioClip:audioData error:nil];
  XCTAssertNotNil(result);

  [MPPAudioClassifierTests assertAudioClassifierResult:result
      approximatelyEqualToExpectedAudioClassifierResult:[MPPAudioClassifierTests
                                                            expectedYamnetInsufficientSilenceResult]
                     expectedClassificationResultsCount:1
           expectedClassificationHeadsCategoryCountInfo:kYamnetModelHeadsInfo];
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

  MPPAudioData *audioClip =
      [MPPAudioClassifierTests audioDataFromAudioFileWithInfo:kSpeech16KHzMonoFileInfo];
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

  MPPAudioData *audioClip =
      [MPPAudioClassifierTests audioDataFromAudioFileWithInfo:kSpeech16KHzMonoFileInfo];

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

#pragma mark Audio Data Initializers

+ (MPPAudioData *)audioDataFromAudioFileWithInfo:(MPPFileInfo *)fileInfo {
  // Load the samples from the audio file in `Float32` interleaved format to
  // an `AVAudioPCMBuffer`.
  AVAudioPCMBuffer *buffer =
      [AVAudioPCMBuffer interleavedFloat32BufferFromAudioFileWithInfo:fileInfo];

  // Create a float buffer from the `floatChannelData` of `AVAudioPCMBuffer`. This float buffer will
  // be used to load the audio data.
  MPPFloatBuffer *bufferData = [[MPPFloatBuffer alloc] initWithData:buffer.floatChannelData[0]
                                                             length:buffer.frameLength];

  // Create the audio data with the same format as the `AVAudioPCMBuffer`.
  MPPAudioDataFormat *audioDataFormat =
      [[MPPAudioDataFormat alloc] initWithChannelCount:buffer.format.channelCount
                                            sampleRate:buffer.format.sampleRate];

  MPPAudioData *audioData = [[MPPAudioData alloc] initWithFormat:audioDataFormat
                                                     sampleCount:buffer.frameLength];

  // Load all the samples in the audio file to the newly created audio data.
  [audioData loadBuffer:bufferData offset:0 length:bufferData.length error:nil];
  return audioData;
}

#pragma mark Audio Classifier Initializers

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
  MPPAudioData *audioData = [MPPAudioClassifierTests audioDataFromAudioFileWithInfo:fileInfo];
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
  return [MPPAudioClassifierTests expectedPartialYamnetResultWithTimestampInMilliseconds:0];
}

// Returns only one top category for each classification head.
// Last classification result (timestamped result) is omitted because it varies between test
// runs due to the low confidence score. Ensure that the subset of classification results in the
// predicted audio classifier result is compared with the expected result returned from this method.
+ (MPPAudioClassifierResult *)expectedPartialYamnetResultWithTimestampInMilliseconds:
    (NSInteger)timestampInMilliseconds {
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

  return [[MPPAudioClassifierResult alloc] initWithClassificationResults:classificationResults
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
