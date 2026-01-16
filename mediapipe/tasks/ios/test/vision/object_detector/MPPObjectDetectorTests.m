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
#import "mediapipe/tasks/ios/vision/object_detector/sources/MPPObjectDetector.h"

static NSString *const kModelName = @"coco_ssd_mobilenet_v1_1.0_quant_2018_06_29";
static NSDictionary *const kCatsAndDogsImage = @{@"name" : @"cats_and_dogs", @"type" : @"jpg"};
static NSDictionary *const kCatsAndDogsRotatedImage =
    @{@"name" : @"cats_and_dogs_rotated", @"type" : @"jpg"};
static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";
static const float pixelDifferenceTolerance = 10.0f;
static const float scoreDifferenceTolerance = 0.02f;
static NSString *const kLiveStreamTestsDictObjectDetectorKey = @"object_detector";
static NSString *const kLiveStreamTestsDictExpectationKey = @"expectation";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

#define AssertEqualCategories(category, expectedCategory, detectionIndex, categoryIndex)           \
  XCTAssertEqual(category.index, expectedCategory.index,                                           \
                 @"detection Index = %d category array index j = %d", detectionIndex,              \
                 categoryIndex);                                                                   \
  XCTAssertEqualWithAccuracy(category.score, expectedCategory.score, scoreDifferenceTolerance,     \
                             @"detection Index = %d, category array index j = %d", detectionIndex, \
                             categoryIndex);                                                       \
  XCTAssertEqualObjects(category.categoryName, expectedCategory.categoryName,                      \
                        @"detection Index = %d, category array index j = %d", detectionIndex,      \
                        categoryIndex);                                                            \
  XCTAssertEqualObjects(category.displayName, expectedCategory.displayName,                        \
                        @"detection Index = %d, category array index j = %d", detectionIndex,      \
                        categoryIndex);

#define AssertApproximatelyEqualBoundingBoxes(boundingBox, expectedBoundingBox, idx)   \
  XCTAssertEqualWithAccuracy(boundingBox.origin.x, expectedBoundingBox.origin.x,       \
                             pixelDifferenceTolerance, @"index i = %d", idx);          \
  XCTAssertEqualWithAccuracy(boundingBox.origin.y, expectedBoundingBox.origin.y,       \
                             pixelDifferenceTolerance, @"index i = %d", idx);          \
  XCTAssertEqualWithAccuracy(boundingBox.size.width, expectedBoundingBox.size.width,   \
                             pixelDifferenceTolerance, @"index i = %d", idx);          \
  XCTAssertEqualWithAccuracy(boundingBox.size.height, expectedBoundingBox.size.height, \
                             pixelDifferenceTolerance, @"index i = %d", idx);

@interface MPPObjectDetectorTests : XCTestCase <MPPObjectDetectorLiveStreamDelegate> {
  NSDictionary *liveStreamSucceedsTestDict;
  NSDictionary *outOfOrderTimestampTestDict;
}
@end

@implementation MPPObjectDetectorTests

#pragma mark Results

+ (MPPObjectDetectorResult *)expectedDetectionResultForCatsAndDogsImageWithTimestampInMilliseconds:
    (NSInteger)timestampInMilliseconds {
  NSArray<MPPDetection *> *detections = @[
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1 score:0.69921875f categoryName:@"cat" displayName:nil],
    ]
                                 boundingBox:CGRectMake(608, 161, 381, 439)
                                   keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1 score:0.656250f categoryName:@"cat" displayName:nil],
    ]
                                 boundingBox:CGRectMake(57, 398, 392, 196)
                                   keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1 score:0.51171875f categoryName:@"cat" displayName:nil],
    ]
                                 boundingBox:CGRectMake(257, 395, 173, 202)
                                   keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1 score:0.48828125f categoryName:@"cat" displayName:nil],
    ]
                                 boundingBox:CGRectMake(363, 195, 330, 412)
                                   keypoints:nil],
  ];

  return [[MPPObjectDetectorResult alloc] initWithDetections:detections
                                     timestampInMilliseconds:timestampInMilliseconds];
}

- (void)assertDetections:(NSArray<MPPDetection *> *)detections
    isEqualToExpectedDetections:(NSArray<MPPDetection *> *)expectedDetections {
  for (int i = 0; i < detections.count; i++) {
    MPPDetection *detection = detections[i];
    XCTAssertNotNil(detection);
    for (int j = 0; j < detection.categories.count; j++) {
      AssertEqualCategories(detection.categories[j], expectedDetections[i].categories[j], i, j);
    }
    AssertApproximatelyEqualBoundingBoxes(detection.boundingBox, expectedDetections[i].boundingBox,
                                          i);
  }
}

- (void)assertObjectDetectorResult:(MPPObjectDetectorResult *)objectDetectorResult
           isEqualToExpectedResult:(MPPObjectDetectorResult *)expectedObjectDetectorResult
           expectedDetectionsCount:(NSInteger)expectedDetectionsCount {
  XCTAssertNotNil(objectDetectorResult);

  NSArray<MPPDetection *> *detectionsSubsetToCompare;
  XCTAssertEqual(objectDetectorResult.detections.count, expectedDetectionsCount);
  if (objectDetectorResult.detections.count > expectedObjectDetectorResult.detections.count) {
    detectionsSubsetToCompare = [objectDetectorResult.detections
        subarrayWithRange:NSMakeRange(0, expectedObjectDetectorResult.detections.count)];
  } else {
    detectionsSubsetToCompare = objectDetectorResult.detections;
  }

  [self assertDetections:detectionsSubsetToCompare
      isEqualToExpectedDetections:expectedObjectDetectorResult.detections];

  XCTAssertEqual(objectDetectorResult.timestampInMilliseconds,
                 expectedObjectDetectorResult.timestampInMilliseconds);
}

#pragma mark File

- (NSString *)filePathWithName:(NSString *)fileName extension:(NSString *)extension {
  NSString *filePath = [[NSBundle bundleForClass:self.class] pathForResource:fileName
                                                                      ofType:extension];
  return filePath;
}

#pragma mark Object Detector Initializers

- (MPPObjectDetectorOptions *)objectDetectorOptionsWithModelName:(NSString *)modelName {
  NSString *modelPath = [self filePathWithName:modelName extension:@"tflite"];
  MPPObjectDetectorOptions *objectDetectorOptions = [[MPPObjectDetectorOptions alloc] init];
  objectDetectorOptions.baseOptions.modelAssetPath = modelPath;

  return objectDetectorOptions;
}

- (void)assertCreateObjectDetectorWithOptions:(MPPObjectDetectorOptions *)objectDetectorOptions
                       failsWithExpectedError:(NSError *)expectedError {
  NSError *error = nil;
  MPPObjectDetector *objectDetector =
      [[MPPObjectDetector alloc] initWithOptions:objectDetectorOptions error:&error];

  XCTAssertNil(objectDetector);
  AssertEqualErrors(error, expectedError);
}

- (MPPObjectDetector *)objectDetectorWithOptionsSucceeds:
    (MPPObjectDetectorOptions *)objectDetectorOptions {
  MPPObjectDetector *objectDetector =
      [[MPPObjectDetector alloc] initWithOptions:objectDetectorOptions error:nil];
  XCTAssertNotNil(objectDetector);

  return objectDetector;
}

#pragma mark Assert Detection Results

- (MPPImage *)imageWithFileInfo:(NSDictionary *)fileInfo {
  MPPImage *image = [MPPImage imageFromBundleWithClass:[MPPObjectDetectorTests class]
                                              fileName:fileInfo[@"name"]
                                                ofType:fileInfo[@"type"]];
  XCTAssertNotNil(image);

  return image;
}

- (MPPImage *)imageWithFileInfo:(NSDictionary *)fileInfo
                    orientation:(UIImageOrientation)orientation {
  MPPImage *image = [MPPImage imageFromBundleWithClass:[MPPObjectDetectorTests class]
                                              fileName:fileInfo[@"name"]
                                                ofType:fileInfo[@"type"]
                                           orientation:orientation];
  XCTAssertNotNil(image);

  return image;
}

- (void)assertResultsOfDetectInImage:(MPPImage *)mppImage
                 usingObjectDetector:(MPPObjectDetector *)objectDetector
                          maxResults:(NSInteger)maxResults
          equalsObjectDetectorResult:(MPPObjectDetectorResult *)expectedObjectDetectorResult {
  MPPObjectDetectorResult *ObjectDetectorResult = [objectDetector detectImage:mppImage error:nil];

  [self assertObjectDetectorResult:ObjectDetectorResult
           isEqualToExpectedResult:expectedObjectDetectorResult
           expectedDetectionsCount:maxResults > 0 ? maxResults
                                                  : ObjectDetectorResult.detections.count];
}

- (void)assertResultsOfDetectInImageWithFileInfo:(NSDictionary *)fileInfo
                             usingObjectDetector:(MPPObjectDetector *)objectDetector
                                      maxResults:(NSInteger)maxResults

                      equalsObjectDetectorResult:
                          (MPPObjectDetectorResult *)expectedObjectDetectorResult {
  MPPImage *mppImage = [self imageWithFileInfo:fileInfo];

  [self assertResultsOfDetectInImage:mppImage
                 usingObjectDetector:objectDetector
                          maxResults:maxResults
          equalsObjectDetectorResult:expectedObjectDetectorResult];
}

#pragma mark General Tests

- (void)testCreateObjectDetectorWithMissingModelPathFails {
  NSString *modelPath = [self filePathWithName:@"" extension:@""];

  NSError *error = nil;
  MPPObjectDetector *objectDetector = [[MPPObjectDetector alloc] initWithModelPath:modelPath
                                                                             error:&error];
  XCTAssertNil(objectDetector);

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

- (void)testCreateObjectDetectorAllowlistAndDenylistFails {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];
  options.categoryAllowlist = @[ @"cat" ];
  options.categoryDenylist = @[ @"dog" ];

  [self assertCreateObjectDetectorWithOptions:options
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

- (void)testDetectWithModelPathSucceeds {
  NSString *modelPath = [self filePathWithName:kModelName extension:@"tflite"];
  MPPObjectDetector *objectDetector = [[MPPObjectDetector alloc] initWithModelPath:modelPath
                                                                             error:nil];
  XCTAssertNotNil(objectDetector);

  [self assertResultsOfDetectInImageWithFileInfo:kCatsAndDogsImage
                             usingObjectDetector:objectDetector
                                      maxResults:-1
                      equalsObjectDetectorResult:
                          [MPPObjectDetectorTests
                              expectedDetectionResultForCatsAndDogsImageWithTimestampInMilliseconds:
                                  0]];
}

- (void)testDetectWithOptionsSucceeds {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  [self assertResultsOfDetectInImageWithFileInfo:kCatsAndDogsImage
                             usingObjectDetector:objectDetector
                                      maxResults:-1
                      equalsObjectDetectorResult:
                          [MPPObjectDetectorTests
                              expectedDetectionResultForCatsAndDogsImageWithTimestampInMilliseconds:
                                  0]];
}

- (void)testDetectWithMaxResultsSucceeds {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  const NSInteger maxResults = 4;
  options.maxResults = maxResults;

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  [self assertResultsOfDetectInImageWithFileInfo:kCatsAndDogsImage
                             usingObjectDetector:objectDetector
                                      maxResults:maxResults
                      equalsObjectDetectorResult:
                          [MPPObjectDetectorTests
                              expectedDetectionResultForCatsAndDogsImageWithTimestampInMilliseconds:
                                  0]];
}

- (void)testDetectWithScoreThresholdSucceeds {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];
  options.scoreThreshold = 0.68f;

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  NSArray<MPPDetection *> *detections = @[
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1 score:0.69921875f categoryName:@"cat" displayName:nil],
    ]
                                 boundingBox:CGRectMake(608, 161, 381, 439)
                                   keypoints:nil],
  ];
  MPPObjectDetectorResult *expectedObjectDetectorResult =
      [[MPPObjectDetectorResult alloc] initWithDetections:detections timestampInMilliseconds:0];

  [self assertResultsOfDetectInImageWithFileInfo:kCatsAndDogsImage
                             usingObjectDetector:objectDetector
                                      maxResults:-1
                      equalsObjectDetectorResult:expectedObjectDetectorResult];
}

- (void)testDetectWithCategoryAllowlistSucceeds {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];
  options.categoryAllowlist = @[ @"cat" ];

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  NSArray<MPPDetection *> *detections = @[
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1 score:0.69921875f categoryName:@"cat" displayName:nil],
    ]
                                 boundingBox:CGRectMake(608, 161, 381, 439)
                                   keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1 score:0.656250f categoryName:@"cat" displayName:nil],
    ]
                                 boundingBox:CGRectMake(57, 398, 392, 196)
                                   keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1 score:0.51171875f categoryName:@"cat" displayName:nil],
    ]
                                 boundingBox:CGRectMake(257, 395, 173, 202)
                                   keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1 score:0.48828125f categoryName:@"cat" displayName:nil],
    ]
                                 boundingBox:CGRectMake(363, 195, 330, 412)
                                   keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1 score:0.355469f categoryName:@"cat" displayName:nil],
    ]
                                 boundingBox:CGRectMake(275, 216, 610, 386)
                                   keypoints:nil],
  ];

  MPPObjectDetectorResult *expectedDetectionResult =
      [[MPPObjectDetectorResult alloc] initWithDetections:detections timestampInMilliseconds:0];

  [self assertResultsOfDetectInImageWithFileInfo:kCatsAndDogsImage
                             usingObjectDetector:objectDetector
                                      maxResults:-1
                      equalsObjectDetectorResult:expectedDetectionResult];
}

- (void)testDetectWithCategoryDenylistSucceeds {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];
  options.categoryDenylist = @[ @"cat" ];

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  NSArray<MPPDetection *> *detections = @[
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1
                                   score:0.476562f
                            categoryName:@"teddy bear"
                             displayName:nil],
    ]
                                 boundingBox:CGRectMake(780, 407, 314, 190)
                                   keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1
                                   score:0.390625f
                            categoryName:@"teddy bear"
                             displayName:nil],
    ]
                                 boundingBox:CGRectMake(90, 225, 568, 366)
                                   keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1
                                   score:0.367188f
                            categoryName:@"teddy bear"
                             displayName:nil],
    ]
                                 boundingBox:CGRectMake(888, 434, 187, 167)
                                   keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1 score:0.332031f categoryName:@"bed" displayName:nil],
    ]
                                 boundingBox:CGRectMake(79, 364, 1097, 224)
                                   keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1
                                   score:0.289062f
                            categoryName:@"teddy bear"
                             displayName:nil],
    ]
                                 boundingBox:CGRectMake(605, 398, 445, 199)
                                   keypoints:nil],
  ];

  MPPObjectDetectorResult *expectedDetectionResult =
      [[MPPObjectDetectorResult alloc] initWithDetections:detections timestampInMilliseconds:0];

  [self assertResultsOfDetectInImageWithFileInfo:kCatsAndDogsImage
                             usingObjectDetector:objectDetector
                                      maxResults:-1
                      equalsObjectDetectorResult:expectedDetectionResult];
}

- (void)testDetectWithOrientationSucceeds {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];
  options.maxResults = 1;

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  NSArray<MPPDetection *> *detections = @[
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1 score:0.699219f categoryName:@"cat" displayName:nil],
    ]
                                 boundingBox:CGRectMake(0, 608, 439, 387)
                                   keypoints:nil],
  ];

  MPPObjectDetectorResult *expectedDetectionResult =
      [[MPPObjectDetectorResult alloc] initWithDetections:detections timestampInMilliseconds:0];

  MPPImage *image = [self imageWithFileInfo:kCatsAndDogsRotatedImage
                                orientation:UIImageOrientationLeft];

  [self assertResultsOfDetectInImage:image
                 usingObjectDetector:objectDetector
                          maxResults:1
          equalsObjectDetectorResult:expectedDetectionResult];
}

#pragma mark Running Mode Tests

- (void)testCreateObjectDetectorFailsWithDelegateInNonLiveStreamMode {
  MPPRunningMode runningModesToTest[] = {MPPRunningModeImage, MPPRunningModeVideo};
  for (int i = 0; i < sizeof(runningModesToTest) / sizeof(runningModesToTest[0]); i++) {
    MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

    options.runningMode = runningModesToTest[i];
    options.objectDetectorLiveStreamDelegate = self;

    [self
        assertCreateObjectDetectorWithOptions:options
                       failsWithExpectedError:
                           [NSError errorWithDomain:kExpectedErrorDomain
                                               code:MPPTasksErrorCodeInvalidArgumentError
                                           userInfo:@{
                                             NSLocalizedDescriptionKey :
                                                 @"The vision task is in image or video mode. The "
                                                 @"delegate must not be set in the task's options."
                                           }]];
  }
}

- (void)testCreateObjectDetectorFailsWithMissingDelegateInLiveStreamMode {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  options.runningMode = MPPRunningModeLiveStream;

  [self assertCreateObjectDetectorWithOptions:options
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

- (void)testDetectFailsWithCallingWrongApiInImageMode {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kCatsAndDogsImage];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([objectDetector detectAsyncImage:image
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
  XCTAssertFalse([objectDetector detectVideoFrame:image
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

- (void)testDetectFailsWithCallingWrongApiInVideoMode {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];
  options.runningMode = MPPRunningModeVideo;

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kCatsAndDogsImage];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([objectDetector detectAsyncImage:image
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
  XCTAssertFalse([objectDetector detectImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Video"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);
}

- (void)testDetectFailsWithCallingWrongApiInLiveStreamMode {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  options.runningMode = MPPRunningModeLiveStream;
  options.objectDetectorLiveStreamDelegate = self;

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kCatsAndDogsImage];

  NSError *imageApiCallError;
  XCTAssertFalse([objectDetector detectImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Live Stream"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);

  NSError *videoApiCallError;
  XCTAssertFalse([objectDetector detectVideoFrame:image
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
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  options.runningMode = MPPRunningModeVideo;

  NSInteger maxResults = 4;
  options.maxResults = maxResults;

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kCatsAndDogsImage];

  for (int i = 0; i < 3; i++) {
    MPPObjectDetectorResult *ObjectDetectorResult = [objectDetector detectVideoFrame:image
                                                             timestampInMilliseconds:i
                                                                               error:nil];

    [self assertObjectDetectorResult:ObjectDetectorResult
             isEqualToExpectedResult:
                 [MPPObjectDetectorTests
                     expectedDetectionResultForCatsAndDogsImageWithTimestampInMilliseconds:i]
             expectedDetectionsCount:maxResults];
  }
}

- (void)testDetectWithOutOfOrderTimestampsAndLiveStreamModeFails {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  NSInteger maxResults = 4;
  options.maxResults = maxResults;

  options.runningMode = MPPRunningModeLiveStream;
  options.objectDetectorLiveStreamDelegate = self;

  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"detectWithOutOfOrderTimestampsAndLiveStream"];
  expectation.expectedFulfillmentCount = 1;

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];
  liveStreamSucceedsTestDict = @{
    kLiveStreamTestsDictObjectDetectorKey : objectDetector,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  MPPImage *image = [self imageWithFileInfo:kCatsAndDogsImage];

  XCTAssertTrue([objectDetector detectAsyncImage:image timestampInMilliseconds:1 error:nil]);

  NSError *error;
  XCTAssertFalse([objectDetector detectAsyncImage:image timestampInMilliseconds:0 error:&error]);

  NSError *expectedError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey :
                            @"INVALID_ARGUMENT: Input timestamp must be monotonically increasing."
                      }];
  AssertEqualErrors(error, expectedError);

  NSTimeInterval timeout = 0.5f;
  [self waitForExpectations:@[ expectation ] timeout:timeout];
}

- (void)testDetectWithLiveStreamModeSucceeds {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  NSInteger maxResults = 4;
  options.maxResults = maxResults;

  options.runningMode = MPPRunningModeLiveStream;

  NSInteger iterationCount = 100;

  // Because of flow limiting, we cannot ensure that the callback will be
  // invoked `iterationCount` times.
  // An normal expectation will fail if expectation.fulfill() is not called
  // `expectation.expectedFulfillmentCount` times.
  // If `expectation.isInverted = true`, the test will only succeed if
  // expectation is not fulfilled for the specified `expectedFulfillmentCount`.
  // Since in our case we cannot predict how many times the expectation is
  // supposed to be fulfilled setting,
  // `expectation.expectedFulfillmentCount` = `iterationCount` + 1 and
  // `expectation.isInverted = true` ensures that test succeeds if
  // expectation is fulfilled <= `iterationCount` times.
  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"detectWithOutOfOrderTimestampsAndLiveStream"];
  expectation.expectedFulfillmentCount = iterationCount + 1;
  expectation.inverted = YES;

  options.objectDetectorLiveStreamDelegate = self;

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  liveStreamSucceedsTestDict = @{
    kLiveStreamTestsDictObjectDetectorKey : objectDetector,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  // TODO: Mimic initialization from CMSampleBuffer as live stream mode is most likely to be used
  // with the iOS camera. AVCaptureVideoDataOutput sample buffer delegates provide frames of type
  // `CMSampleBuffer`.
  MPPImage *image = [self imageWithFileInfo:kCatsAndDogsImage];

  for (int i = 0; i < iterationCount; i++) {
    XCTAssertTrue([objectDetector detectAsyncImage:image timestampInMilliseconds:i error:nil]);
  }

  NSTimeInterval timeout = 0.5f;
  [self waitForExpectations:@[ expectation ] timeout:timeout];
}

#pragma mark MPPObjectDetectorLiveStreamDelegate Methods
- (void)objectDetector:(MPPObjectDetector *)objectDetector
    didFinishDetectionWithResult:(MPPObjectDetectorResult *)ObjectDetectorResult
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(NSError *)error {
  NSInteger maxResults = 4;
  [self assertObjectDetectorResult:ObjectDetectorResult
           isEqualToExpectedResult:
               [MPPObjectDetectorTests
                   expectedDetectionResultForCatsAndDogsImageWithTimestampInMilliseconds:
                       timestampInMilliseconds]
           expectedDetectionsCount:maxResults];

  if (objectDetector == outOfOrderTimestampTestDict[kLiveStreamTestsDictObjectDetectorKey]) {
    [outOfOrderTimestampTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  } else if (objectDetector == liveStreamSucceedsTestDict[kLiveStreamTestsDictObjectDetectorKey]) {
    [liveStreamSucceedsTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  }
}

@end
