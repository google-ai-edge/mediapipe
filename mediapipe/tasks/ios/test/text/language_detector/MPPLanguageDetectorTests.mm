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

#import <XCTest/XCTest.h>

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/test/utils/sources/MPPFileInfo.h"
#import "mediapipe/tasks/ios/text/language_detector/sources/MPPLanguageDetector.h"

static MPPFileInfo *const kLanguageDetectorModelFileInfo =
    [[MPPFileInfo alloc] initWithName:@"language_detector" type:@"tflite"];

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

@interface MPPLanguageDetectorTests : XCTestCase
@end

@implementation MPPLanguageDetectorTests

- (void)testCreateLanguageDetectorFailsWithMissingModelPath {
  MPPFileInfo *fileInfo = [[MPPFileInfo alloc] initWithName:@"" type:@""];

  NSError *error = nil;
  MPPLanguageDetector *languageDetector =
      [[MPPLanguageDetector alloc] initWithModelPath:fileInfo.path error:&error];
  XCTAssertNil(languageDetector);

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

- (void)testCreateLanguageDetectorFailsWithBothAllowlistAndDenylist {
  MPPLanguageDetectorOptions *options =
      [self languageDetectorOptionsWithModelFileInfo:kLanguageDetectorModelFileInfo];
  options.categoryAllowlist = @[ @"en" ];
  options.categoryDenylist = @[ @"en" ];

  [self assertCreateLanguageDetectorWithOptions:options
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

- (void)testCreateLanguageDetectorFailsWithInvalidMaxResults {
  MPPLanguageDetectorOptions *options =
      [self languageDetectorOptionsWithModelFileInfo:kLanguageDetectorModelFileInfo];
  options.maxResults = 0;

  [self
      assertCreateLanguageDetectorWithOptions:options
                       failsWithExpectedError:
                           [NSError errorWithDomain:kExpectedErrorDomain
                                               code:MPPTasksErrorCodeInvalidArgumentError
                                           userInfo:@{
                                             NSLocalizedDescriptionKey :
                                                 @"INVALID_ARGUMENT: Invalid `max_results` option: "
                                                 @"value must be != 0."
                                           }]];
}

- (void)testClassifyWithL2CModelSucceeds {
  MPPLanguageDetectorOptions *options =
      [self languageDetectorOptionsWithModelFileInfo:kLanguageDetectorModelFileInfo];

  MPPLanguageDetector *languageDetector = [self createLanguageDetectorWithOptionsSucceeds:options];
  NSString *enText = @"To be, or not to be, that is the question";
  NSArray<MPPLanguagePrediction *> *expectedEnLanguagePredictions =
      @[ [[MPPLanguagePrediction alloc] initWithLanguageCode:@"en" probability:0.9998559f] ];

  [self assertResultsOfDetectLanguageOfText:enText
                               usingLanguageDetector:languageDetector
      approximatelyEqualsExpectedLanguagePredictions:expectedEnLanguagePredictions];

  NSString *frText = @"Il y a beaucoup de bouches qui parlent et fort peu de têtes qui pensent.";
  NSArray<MPPLanguagePrediction *> *expectedFrLanguagePredictions =
      @[ [[MPPLanguagePrediction alloc] initWithLanguageCode:@"fr" probability:0.9997813f] ];

  [self assertResultsOfDetectLanguageOfText:frText
                               usingLanguageDetector:languageDetector
      approximatelyEqualsExpectedLanguagePredictions:expectedFrLanguagePredictions];

  NSString *ruText = @"это какой-то английский язык";
  NSArray<MPPLanguagePrediction *> *expectedRuLanguagePredictions =
      @[ [[MPPLanguagePrediction alloc] initWithLanguageCode:@"ru" probability:0.9933616f] ];

  [self assertResultsOfDetectLanguageOfText:ruText
                               usingLanguageDetector:languageDetector
      approximatelyEqualsExpectedLanguagePredictions:expectedRuLanguagePredictions];

  NSString *zhText = @"分久必合合久必分";
  NSArray<MPPLanguagePrediction *> *expectedZhLanguagePredictions = @[
    [[MPPLanguagePrediction alloc] initWithLanguageCode:@"zh" probability:0.505424f],
    [[MPPLanguagePrediction alloc] initWithLanguageCode:@"ja" probability:0.481617f]
  ];

  [self assertResultsOfDetectLanguageOfText:zhText
                               usingLanguageDetector:languageDetector
      approximatelyEqualsExpectedLanguagePredictions:expectedZhLanguagePredictions];
}

- (void)testClassifyWithMaxResultsSucceeds {
  MPPLanguageDetectorOptions *options =
      [self languageDetectorOptionsWithModelFileInfo:kLanguageDetectorModelFileInfo];
  options.maxResults = 1;
  MPPLanguageDetector *languageDetector = [self createLanguageDetectorWithOptionsSucceeds:options];

  NSString *zhText = @"分久必合合久必分";
  NSArray<MPPLanguagePrediction *> *expectedZhLanguagePredictions = @[
    [[MPPLanguagePrediction alloc] initWithLanguageCode:@"zh" probability:0.505424f],
  ];

  [self assertResultsOfDetectLanguageOfText:zhText
                               usingLanguageDetector:languageDetector
      approximatelyEqualsExpectedLanguagePredictions:expectedZhLanguagePredictions];
}

- (void)testClassifyWithScoreThresholdSucceeds {
  MPPLanguageDetectorOptions *options =
      [self languageDetectorOptionsWithModelFileInfo:kLanguageDetectorModelFileInfo];
  options.scoreThreshold = 0.5f;
  MPPLanguageDetector *languageDetector = [self createLanguageDetectorWithOptionsSucceeds:options];

  NSString *zhText = @"分久必合合久必分";
  NSArray<MPPLanguagePrediction *> *expectedZhLanguagePredictions = @[
    [[MPPLanguagePrediction alloc] initWithLanguageCode:@"zh" probability:0.505424f],
  ];

  [self assertResultsOfDetectLanguageOfText:zhText
                               usingLanguageDetector:languageDetector
      approximatelyEqualsExpectedLanguagePredictions:expectedZhLanguagePredictions];
}

- (void)testClassifyWithCategoryAllowListSucceeds {
  MPPLanguageDetectorOptions *options =
      [self languageDetectorOptionsWithModelFileInfo:kLanguageDetectorModelFileInfo];
  options.categoryAllowlist = @[ @"zh" ];

  MPPLanguageDetector *languageDetector = [self createLanguageDetectorWithOptionsSucceeds:options];

  NSString *zhText = @"分久必合合久必分";
  NSArray<MPPLanguagePrediction *> *expectedZhLanguagePredictions = @[
    [[MPPLanguagePrediction alloc] initWithLanguageCode:@"zh" probability:0.505424f],
  ];

  [self assertResultsOfDetectLanguageOfText:zhText
                               usingLanguageDetector:languageDetector
      approximatelyEqualsExpectedLanguagePredictions:expectedZhLanguagePredictions];
}

- (void)testClassifyWithCategoryDenyListSucceeds {
  MPPLanguageDetectorOptions *options =
      [self languageDetectorOptionsWithModelFileInfo:kLanguageDetectorModelFileInfo];
  options.categoryDenylist = @[ @"zh" ];

  MPPLanguageDetector *languageDetector = [self createLanguageDetectorWithOptionsSucceeds:options];

  NSString *zhText = @"分久必合合久必分";
  NSArray<MPPLanguagePrediction *> *expectedZhLanguagePredictions = @[
    [[MPPLanguagePrediction alloc] initWithLanguageCode:@"ja" probability:0.481617f],
  ];

  [self assertResultsOfDetectLanguageOfText:zhText
                               usingLanguageDetector:languageDetector
      approximatelyEqualsExpectedLanguagePredictions:expectedZhLanguagePredictions];
}

#pragma mark Assert Segmenter Results
- (void)assertResultsOfDetectLanguageOfText:(NSString *)text
                             usingLanguageDetector:(MPPLanguageDetector *)languageDetector
    approximatelyEqualsExpectedLanguagePredictions:
        (NSArray<MPPLanguagePrediction *> *)expectedLanguagePredictions {
  MPPLanguageDetectorResult *result = [languageDetector detectText:text error:nil];
  XCTAssertNotNil(result);

  XCTAssertEqual(result.languagePredictions.count, expectedLanguagePredictions.count);
  XCTAssertEqualWithAccuracy(result.languagePredictions[0].probability,
                             expectedLanguagePredictions[0].probability, 1e-3);
  XCTAssertEqualObjects(result.languagePredictions[0].languageCode,
                        expectedLanguagePredictions[0].languageCode);
}

#pragma mark Language Detector Initializers

- (MPPLanguageDetectorOptions *)languageDetectorOptionsWithModelFileInfo:(MPPFileInfo *)fileInfo {
  MPPLanguageDetectorOptions *options = [[MPPLanguageDetectorOptions alloc] init];
  options.baseOptions.modelAssetPath = fileInfo.path;
  return options;
}

- (MPPLanguageDetector *)createLanguageDetectorWithOptionsSucceeds:
    (MPPLanguageDetectorOptions *)options {
  NSError *error;
  MPPLanguageDetector *languageDetector = [[MPPLanguageDetector alloc] initWithOptions:options
                                                                                 error:&error];
  XCTAssertNotNil(languageDetector);
  XCTAssertNil(error);

  return languageDetector;
}

- (void)assertCreateLanguageDetectorWithOptions:(MPPLanguageDetectorOptions *)options
                         failsWithExpectedError:(NSError *)expectedError {
  NSError *error = nil;
  MPPLanguageDetector *languageDetector = [[MPPLanguageDetector alloc] initWithOptions:options
                                                                                 error:&error];
  XCTAssertNil(languageDetector);
  AssertEqualErrors(error, expectedError);
}

@end
