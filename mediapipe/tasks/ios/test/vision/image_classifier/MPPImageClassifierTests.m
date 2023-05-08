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
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"
#import "mediapipe/tasks/ios/vision/image_classifier/sources/MPPImageClassifier.h"

static NSString *kFloatModelName = @"mobilenet_v2_1.0_224";
static NSString *const kQuantizedModelName = @"mobilenet_v1_0.25_224_quant";
static NSDictionary *const kBurgerImage = @{@"name" : @"burger", @"type" : @"jpg"};
static NSDictionary *const kBurgerRotatedImage = @{@"name" : @"burger_rotated", @"type" : @"jpg"};
static NSDictionary *const kMultiObjectsImage = @{@"name" : @"multi_objects", @"type" : @"jpg"};
static NSDictionary *const kMultiObjectsRotatedImage =
    @{@"name" : @"multi_objects_rotated", @"type" : @"jpg"};
static const int kMobileNetCategoriesCount = 1001;
static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";

#define AssertEqualErrors(error, expectedError)                                               \
  XCTAssertNotNil(error);                                                                     \
  XCTAssertEqualObjects(error.domain, expectedError.domain);                                  \
  XCTAssertEqual(error.code, expectedError.code);                                             \
  XCTAssertNotEqual(                                                                          \
      [error.localizedDescription rangeOfString:expectedError.localizedDescription].location, \
      NSNotFound)

#define AssertEqualCategoryArrays(categories, expectedCategories)                         \
  XCTAssertEqual(categories.count, expectedCategories.count);                             \
  for (int i = 0; i < categories.count; i++) {                                            \
    XCTAssertEqual(categories[i].index, expectedCategories[i].index, @"index i = %d", i); \
    XCTAssertEqualWithAccuracy(categories[i].score, expectedCategories[i].score, 1e-2,    \
                               @"index i = %d", i);                                       \
    XCTAssertEqualObjects(categories[i].categoryName, expectedCategories[i].categoryName, \
                          @"index i = %d", i);                                            \
    XCTAssertEqualObjects(categories[i].displayName, expectedCategories[i].displayName,   \
                          @"index i = %d", i);                                            \
  }

#define AssertImageClassifierResultHasOneHead(imageClassifierResult)                   \
  XCTAssertNotNil(imageClassifierResult);                                              \
  XCTAssertNotNil(imageClassifierResult.classificationResult);                         \
  XCTAssertEqual(imageClassifierResult.classificationResult.classifications.count, 1); \
  XCTAssertEqual(imageClassifierResult.classificationResult.classifications[0].headIndex, 0);

@interface MPPImageClassifierTests : XCTestCase
@end

@implementation MPPImageClassifierTests

#pragma mark Results

+ (NSArray<MPPCategory *> *)expectedResultCategoriesForClassifyBurgerImageWithFloatModel {
  return @[
    [[MPPCategory alloc] initWithIndex:934
                                 score:0.786005f
                          categoryName:@"cheeseburger"
                           displayName:nil],
    [[MPPCategory alloc] initWithIndex:932 score:0.023508f categoryName:@"bagel" displayName:nil],
    [[MPPCategory alloc] initWithIndex:925
                                 score:0.021172f
                          categoryName:@"guacamole"
                           displayName:nil]
  ];
}

#pragma mark File

- (NSString *)filePathWithName:(NSString *)fileName extension:(NSString *)extension {
  NSString *filePath = [[NSBundle bundleForClass:self.class] pathForResource:fileName
                                                                      ofType:extension];
  return filePath;
}

#pragma mark Classifier Initializers

- (MPPImageClassifierOptions *)imageClassifierOptionsWithModelName:(NSString *)modelName {
  NSString *modelPath = [self filePathWithName:modelName extension:@"tflite"];
  MPPImageClassifierOptions *imageClassifierOptions = [[MPPImageClassifierOptions alloc] init];
  imageClassifierOptions.baseOptions.modelAssetPath = modelPath;

  return imageClassifierOptions;
}

- (MPPImageClassifier *)imageClassifierFromModelFileWithName:(NSString *)modelName {
  NSString *modelPath = [self filePathWithName:modelName extension:@"tflite"];
  MPPImageClassifier *imageClassifier = [[MPPImageClassifier alloc] initWithModelPath:modelPath
                                                                                error:nil];
  XCTAssertNotNil(imageClassifier);

  return imageClassifier;
}

- (MPPImageClassifier *)imageClassifierWithOptionsSucceeds:
    (MPPImageClassifierOptions *)imageClassifierOptions {
  MPPImageClassifier *imageClassifier =
      [[MPPImageClassifier alloc] initWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  return imageClassifier;
}

#pragma mark Assert Classify Results

- (MPPImage *)imageWithFileInfo:(NSDictionary *)fileInfo {
  MPPImage *image = [MPPImage imageFromBundleWithClass:[MPPImageClassifierTests class]
                                              fileName:fileInfo[@"name"]
                                                ofType:fileInfo[@"type"]];
  XCTAssertNotNil(image);

  return image;
}

- (MPPImage *)imageWithFileInfo:(NSDictionary *)fileInfo
                    orientation:(UIImageOrientation)orientation {
  MPPImage *image = [MPPImage imageFromBundleWithClass:[MPPImageClassifierTests class]
                                              fileName:fileInfo[@"name"]
                                                ofType:fileInfo[@"type"]
                                           orientation:orientation];
  XCTAssertNotNil(image);

  return image;
}

- (void)assertCreateImageClassifierWithOptions:(MPPImageClassifierOptions *)imageClassifierOptions
                        failsWithExpectedError:(NSError *)expectedError {
  NSError *error = nil;
  MPPImageClassifier *imageClassifier =
      [[MPPImageClassifier alloc] initWithOptions:imageClassifierOptions error:&error];

  XCTAssertNil(imageClassifier);
  AssertEqualErrors(error, expectedError);
}

- (void)assertImageClassifierResult:(MPPImageClassifierResult *)imageClassifierResult
         hasExpectedCategoriesCount:(NSInteger)expectedCategoriesCount
                 expectedCategories:(NSArray<MPPCategory *> *)expectedCategories {
  AssertImageClassifierResultHasOneHead(imageClassifierResult);

  NSArray<MPPCategory *> *resultCategories =
      imageClassifierResult.classificationResult.classifications[0].categories;
  XCTAssertEqual(resultCategories.count, expectedCategoriesCount);

  NSArray<MPPCategory *> *categorySubsetToCompare;
  if (resultCategories.count > expectedCategories.count) {
    categorySubsetToCompare =
        [resultCategories subarrayWithRange:NSMakeRange(0, expectedCategories.count)];
  } else {
    categorySubsetToCompare = resultCategories;
  }
  AssertEqualCategoryArrays(categorySubsetToCompare, expectedCategories);
}

- (void)assertResultsOfClassifyImage:(MPPImage *)mppImage
                usingImageClassifier:(MPPImageClassifier *)imageClassifier
             expectedCategoriesCount:(NSInteger)expectedCategoriesCount
                    equalsCategories:(NSArray<MPPCategory *> *)expectedCategories {
  MPPImageClassifierResult *imageClassifierResult = [imageClassifier classifyImage:mppImage
                                                                             error:nil];

  [self assertImageClassifierResult:imageClassifierResult
         hasExpectedCategoriesCount:expectedCategoriesCount
                 expectedCategories:expectedCategories];
}

- (void)assertResultsOfClassifyImageWithFileInfo:(NSDictionary *)fileInfo
                            usingImageClassifier:(MPPImageClassifier *)imageClassifier
                         expectedCategoriesCount:(NSInteger)expectedCategoriesCount
                                equalsCategories:(NSArray<MPPCategory *> *)expectedCategories {
  MPPImage *mppImage = [self imageWithFileInfo:fileInfo];

  [self assertResultsOfClassifyImage:mppImage
                usingImageClassifier:imageClassifier
             expectedCategoriesCount:expectedCategoriesCount
                    equalsCategories:expectedCategories];
}

#pragma mark General Tests

- (void)testCreateImageClassifierWithMissingModelPathFails {
  NSString *modelPath = [self filePathWithName:@"" extension:@""];

  NSError *error = nil;
  MPPImageClassifier *imageClassifier = [[MPPImageClassifier alloc] initWithModelPath:modelPath
                                                                                error:&error];
  XCTAssertNil(imageClassifier);

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

- (void)testCreateImageClassifierAllowlistAndDenylistFails {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];
  options.categoryAllowlist = @[ @"cheeseburger" ];
  options.categoryDenylist = @[ @"bagel" ];

  [self assertCreateImageClassifierWithOptions:options
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

- (void)testClassifyWithModelPathAndFloatModelSucceeds {
  MPPImageClassifier *imageClassifier = [self imageClassifierFromModelFileWithName:kFloatModelName];

  [self
      assertResultsOfClassifyImageWithFileInfo:kBurgerImage
                          usingImageClassifier:imageClassifier
                       expectedCategoriesCount:kMobileNetCategoriesCount
                              equalsCategories:
                                  [MPPImageClassifierTests
                                      expectedResultCategoriesForClassifyBurgerImageWithFloatModel]];
}

- (void)testClassifyWithOptionsAndFloatModelSucceeds {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  const NSInteger maxResults = 3;
  options.maxResults = maxResults;

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  [self
      assertResultsOfClassifyImageWithFileInfo:kBurgerImage
                          usingImageClassifier:imageClassifier
                       expectedCategoriesCount:maxResults
                              equalsCategories:
                                  [MPPImageClassifierTests
                                      expectedResultCategoriesForClassifyBurgerImageWithFloatModel]];
}

- (void)testClassifyWithQuantizedModelSucceeds {
  MPPImageClassifierOptions *options =
      [self imageClassifierOptionsWithModelName:kQuantizedModelName];

  const NSInteger maxResults = 1;
  options.maxResults = maxResults;

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  NSArray<MPPCategory *> *expectedCategories = @[ [[MPPCategory alloc] initWithIndex:934
                                                                               score:0.968750f
                                                                        categoryName:@"cheeseburger"
                                                                         displayName:nil] ];

  [self assertResultsOfClassifyImageWithFileInfo:kBurgerImage
                            usingImageClassifier:imageClassifier
                         expectedCategoriesCount:maxResults
                                equalsCategories:expectedCategories];
}

- (void)testClassifyWithScoreThresholdSucceeds {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  options.scoreThreshold = 0.25f;

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  NSArray<MPPCategory *> *expectedCategories = @[ [[MPPCategory alloc] initWithIndex:934
                                                                               score:0.786005f
                                                                        categoryName:@"cheeseburger"
                                                                         displayName:nil] ];

  [self assertResultsOfClassifyImageWithFileInfo:kBurgerImage
                            usingImageClassifier:imageClassifier
                         expectedCategoriesCount:expectedCategories.count
                                equalsCategories:expectedCategories];
}

- (void)testClassifyWithAllowlistSucceeds {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  options.categoryAllowlist = @[ @"cheeseburger", @"guacamole", @"meat loaf" ];

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  NSArray<MPPCategory *> *expectedCategories = @[
    [[MPPCategory alloc] initWithIndex:934
                                 score:0.786005f
                          categoryName:@"cheeseburger"
                           displayName:nil],
    [[MPPCategory alloc] initWithIndex:925
                                 score:0.021172f
                          categoryName:@"guacamole"
                           displayName:nil],
    [[MPPCategory alloc] initWithIndex:963
                                 score:0.006279315f
                          categoryName:@"meat loaf"
                           displayName:nil],

  ];

  [self assertResultsOfClassifyImageWithFileInfo:kBurgerImage
                            usingImageClassifier:imageClassifier
                         expectedCategoriesCount:expectedCategories.count
                                equalsCategories:expectedCategories];
}

- (void)testClassifyWithDenylistSucceeds {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  options.categoryDenylist = @[
    @"bagel",
  ];
  options.maxResults = 3;

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  NSArray<MPPCategory *> *expectedCategories = @[
    [[MPPCategory alloc] initWithIndex:934
                                 score:0.786005f
                          categoryName:@"cheeseburger"
                           displayName:nil],
    [[MPPCategory alloc] initWithIndex:925
                                 score:0.021172f
                          categoryName:@"guacamole"
                           displayName:nil],
    [[MPPCategory alloc] initWithIndex:963
                                 score:0.006279315f
                          categoryName:@"meat loaf"
                           displayName:nil],

  ];

  [self assertResultsOfClassifyImageWithFileInfo:kBurgerImage
                            usingImageClassifier:imageClassifier
                         expectedCategoriesCount:expectedCategories.count
                                equalsCategories:expectedCategories];
}

- (void)testClassifyWithRegionOfInterestSucceeds {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  NSInteger maxResults = 1;
  options.maxResults = maxResults;

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  NSArray<MPPCategory *> *expectedCategories = @[ [[MPPCategory alloc] initWithIndex:806
                                                                               score:0.997122f
                                                                        categoryName:@"soccer ball"
                                                                         displayName:nil] ];

  MPPImage *image = [self imageWithFileInfo:kMultiObjectsImage];

  // roi around soccer ball
  MPPImageClassifierResult *imageClassifierResult =
      [imageClassifier classifyImage:image
                    regionOfInterest:CGRectMake(0.450f, 0.308f, 0.164f, 0.426f)
                               error:nil];
  [self assertImageClassifierResult:imageClassifierResult
         hasExpectedCategoriesCount:maxResults
                 expectedCategories:expectedCategories];
}

- (void)testClassifyWithOrientationSucceeds {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  NSInteger maxResults = 3;
  options.maxResults = maxResults;

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  NSArray<MPPCategory *> *expectedCategories = @[
    [[MPPCategory alloc] initWithIndex:934
                                 score:0.753852f
                          categoryName:@"cheeseburger"
                           displayName:nil],
    [[MPPCategory alloc] initWithIndex:925
                                 score:0.028609f
                          categoryName:@"guacamole"
                           displayName:nil],
    [[MPPCategory alloc] initWithIndex:932 score:0.027782f categoryName:@"bagel" displayName:nil]

  ];

  MPPImage *image = [self imageWithFileInfo:kBurgerRotatedImage
                                orientation:UIImageOrientationRight];

  [self assertResultsOfClassifyImage:image
                usingImageClassifier:imageClassifier
             expectedCategoriesCount:maxResults
                    equalsCategories:expectedCategories];
}

- (void)testClassifyWithRegionOfInterestAndOrientationSucceeds {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  NSInteger maxResults = 1;
  options.maxResults = maxResults;

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  NSArray<MPPCategory *> *expectedCategories =
      @[ [[MPPCategory alloc] initWithIndex:560
                                      score:0.604605f
                               categoryName:@"folding chair"
                                displayName:nil] ];

  MPPImage *image = [self imageWithFileInfo:kMultiObjectsRotatedImage
                                orientation:UIImageOrientationRight];

  // roi around folding chair
  MPPImageClassifierResult *imageClassifierResult =
      [imageClassifier classifyImage:image
                    regionOfInterest:CGRectMake(0.0f, 0.1763f, 0.5642f, 0.1286f)
                               error:nil];
  [self assertImageClassifierResult:imageClassifierResult
         hasExpectedCategoriesCount:maxResults
                 expectedCategories:expectedCategories];
}

#pragma mark Running Mode Tests

- (void)testCreateImageClassifierFailsWithResultListenerInNonLiveStreamMode {
  MPPRunningMode runningModesToTest[] = {MPPRunningModeImage, MPPRunningModeVideo};
  for (int i = 0; i < sizeof(runningModesToTest) / sizeof(runningModesToTest[0]); i++) {
    MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

    options.runningMode = runningModesToTest[i];
    options.completion = ^(MPPImageClassifierResult *result, NSError *error) {
    };

    [self
        assertCreateImageClassifierWithOptions:options
                        failsWithExpectedError:
                            [NSError
                                errorWithDomain:kExpectedErrorDomain
                                           code:MPPTasksErrorCodeInvalidArgumentError
                                       userInfo:@{
                                         NSLocalizedDescriptionKey :
                                             @"The vision task is in image or video mode. The "
                                             @"delegate must not be set in the task's options."
                                       }]];
  }
}

- (void)testCreateImageClassifierFailsWithMissingResultListenerInLiveStreamMode {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  options.runningMode = MPPRunningModeLiveStream;

  [self assertCreateImageClassifierWithOptions:options
                        failsWithExpectedError:
                            [NSError errorWithDomain:kExpectedErrorDomain
                                                code:MPPTasksErrorCodeInvalidArgumentError
                                            userInfo:@{
                                              NSLocalizedDescriptionKey :
                                                  @"The vision task is in live stream mode. An "
                                                  @"object must be set as the delegate of the task "
                                                  @"in its options to ensure asynchronous delivery "
                                                  @"of results."
                                            }]];
}

- (void)testClassifyFailsWithCallingWrongApiInImageMode {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kBurgerImage];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([imageClassifier classifyAsyncImage:image
                             timestampInMilliseconds:0
                                               error:&liveStreamApiCallError]);

  NSError *expectedLiveStreamApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with live "
                                                    @"stream mode. Current Running Mode: Image"
                      }];

  AssertEqualErrors(liveStreamApiCallError, expectedLiveStreamApiCallError);

  NSError *videoApiCallError;
  XCTAssertFalse([imageClassifier classifyVideoFrame:image
                             timestampInMilliseconds:0
                                               error:&videoApiCallError]);

  NSError *expectedVideoApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"video mode. Current Running Mode: Image"
                      }];
  AssertEqualErrors(videoApiCallError, expectedVideoApiCallError);
}

- (void)testClassifyFailsWithCallingWrongApiInVideoMode {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  options.runningMode = MPPRunningModeVideo;

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kBurgerImage];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([imageClassifier classifyAsyncImage:image
                             timestampInMilliseconds:0
                                               error:&liveStreamApiCallError]);

  NSError *expectedLiveStreamApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with live "
                                                    @"stream mode. Current Running Mode: Video"
                      }];

  AssertEqualErrors(liveStreamApiCallError, expectedLiveStreamApiCallError);

  NSError *imageApiCallError;
  XCTAssertFalse([imageClassifier classifyImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Video"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);
}

- (void)testClassifyFailsWithCallingWrongApiInLiveStreamMode {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  options.runningMode = MPPRunningModeLiveStream;
  options.completion = ^(MPPImageClassifierResult *result, NSError *error) {

  };

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kBurgerImage];

  NSError *imageApiCallError;
  XCTAssertFalse([imageClassifier classifyImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Live Stream"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);

  NSError *videoApiCallError;
  XCTAssertFalse([imageClassifier classifyVideoFrame:image
                             timestampInMilliseconds:0
                                               error:&videoApiCallError]);

  NSError *expectedVideoApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"video mode. Current Running Mode: Live Stream"
                      }];
  AssertEqualErrors(videoApiCallError, expectedVideoApiCallError);
}

- (void)testClassifyWithVideoModeSucceeds {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  options.runningMode = MPPRunningModeVideo;

  NSInteger maxResults = 3;
  options.maxResults = maxResults;

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kBurgerImage];

  for (int i = 0; i < 3; i++) {
    MPPImageClassifierResult *imageClassifierResult = [imageClassifier classifyVideoFrame:image
                                                                  timestampInMilliseconds:i
                                                                                    error:nil];
    [self assertImageClassifierResult:imageClassifierResult
           hasExpectedCategoriesCount:maxResults
                   expectedCategories:
                       [MPPImageClassifierTests
                           expectedResultCategoriesForClassifyBurgerImageWithFloatModel]];
  }
}

- (void)testClassifyWithOutOfOrderTimestampsAndLiveStreamModeFails {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  NSInteger maxResults = 3;
  options.maxResults = maxResults;

  options.runningMode = MPPRunningModeLiveStream;
  options.completion = ^(MPPImageClassifierResult *result, NSError *error) {
    [self assertImageClassifierResult:result
           hasExpectedCategoriesCount:maxResults
                   expectedCategories:
                       [MPPImageClassifierTests
                           expectedResultCategoriesForClassifyBurgerImageWithFloatModel]];
  };

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kBurgerImage];

  XCTAssertTrue([imageClassifier classifyAsyncImage:image timestampInMilliseconds:1 error:nil]);

  NSError *error;
  XCTAssertFalse([imageClassifier classifyAsyncImage:image timestampInMilliseconds:0 error:&error]);

  NSError *expectedError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey :
                            @"INVALID_ARGUMENT: Input timestamp must be monotonically increasing."
                      }];
  AssertEqualErrors(error, expectedError);
}

- (void)testClassifyWithLiveStreamModeSucceeds {
  MPPImageClassifierOptions *options = [self imageClassifierOptionsWithModelName:kFloatModelName];

  NSInteger maxResults = 3;
  options.maxResults = maxResults;

  options.runningMode = MPPRunningModeLiveStream;
  options.completion = ^(MPPImageClassifierResult *result, NSError *error) {
    [self assertImageClassifierResult:result
           hasExpectedCategoriesCount:maxResults
                   expectedCategories:
                       [MPPImageClassifierTests
                           expectedResultCategoriesForClassifyBurgerImageWithFloatModel]];
  };

  MPPImageClassifier *imageClassifier = [self imageClassifierWithOptionsSucceeds:options];

  // TODO: Mimic initialization from CMSampleBuffer as live stream mode is most likely to be used
  // with the iOS camera. AVCaptureVideoDataOutput sample buffer delegates provide frames of type
  // `CMSampleBuffer`.
  MPPImage *image = [self imageWithFileInfo:kBurgerImage];

  for (int i = 0; i < 3; i++) {
    XCTAssertTrue([imageClassifier classifyAsyncImage:image timestampInMilliseconds:i error:nil]);
  }
}

@end
