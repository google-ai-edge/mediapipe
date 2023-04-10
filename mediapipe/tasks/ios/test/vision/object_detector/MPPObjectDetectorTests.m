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

#define PixelDifferenceTolerance 5.0f
#define ScoreDifferenceTolerance 1e-2f

#define AssertEqualErrors(error, expectedError)                                               \
  XCTAssertNotNil(error);                                                                     \
  XCTAssertEqualObjects(error.domain, expectedError.domain);                                  \
  XCTAssertEqual(error.code, expectedError.code);                                             \
  XCTAssertNotEqual(                                                                          \
      [error.localizedDescription rangeOfString:expectedError.localizedDescription].location, \
      NSNotFound)

#define AssertEqualCategoryArrays(categories, expectedCategories, detectionIndex)               \
  XCTAssertEqual(categories.count, expectedCategories.count);                                   \
  for (int j = 0; j < categories.count; j++) {                                                  \
    XCTAssertEqual(categories[j].index, expectedCategories[j].index,                            \
                   @"detection Index = %d category array index j = %d", detectionIndex, j);     \
    XCTAssertEqualWithAccuracy(                                                                 \
        categories[j].score, expectedCategories[j].score, ScoreDifferenceTolerance,             \
        @"detection Index = %d, category array index j = %d", detectionIndex, j);               \
    XCTAssertEqualObjects(categories[j].categoryName, expectedCategories[j].categoryName,       \
                          @"detection Index = %d, category array index j = %d", detectionIndex, \
                          j);                                                                   \
    XCTAssertEqualObjects(categories[j].displayName, expectedCategories[j].displayName,         \
                          @"detection Index = %d, category array index j = %d", detectionIndex, \
                          j);                                                                   \
    \                           
                                                                                                                                                                                                                                                                                                                                  \
  }

#define AssertApproximatelyEqualBoundingBoxes(boundingBox, expectedBoundingBox, idx)   \
  XCTAssertEqualWithAccuracy(boundingBox.origin.x, expectedBoundingBox.origin.x,       \
                             PixelDifferenceTolerance, @"index i = %d", idx);          \
  XCTAssertEqualWithAccuracy(boundingBox.origin.y, expectedBoundingBox.origin.y,       \
                             PixelDifferenceTolerance, @"index i = %d", idx);          \
  XCTAssertEqualWithAccuracy(boundingBox.size.width, expectedBoundingBox.size.width,   \
                             PixelDifferenceTolerance, @"index i = %d", idx);          \
  XCTAssertEqualWithAccuracy(boundingBox.size.height, expectedBoundingBox.size.height, \
                             PixelDifferenceTolerance, @"index i = %d", idx);

#define AssertEqualDetections(detection, expectedDetection, idx)                      \
  XCTAssertNotNil(detection);                                                         \
  AssertEqualCategoryArrays(detection.categories, expectedDetection.categories, idx); \
  AssertApproximatelyEqualBoundingBoxes(detection.boundingBox, expectedDetection.boundingBox, idx);

#define AssertEqualDetectionArrays(detections, expectedDetections)  \
  XCTAssertEqual(detections.count, expectedDetections.count);       \
  for (int i = 0; i < detections.count; i++) {                      \
    AssertEqualDetections(detections[i], expectedDetections[i], i); \
  }

#define AssertEqualObjectDetectionResults(objectDetectionResult, expectedObjectDetectionResult,    \
                                          expectedDetectionCount)                                  \
  XCTAssertNotNil(objectDetectionResult);                                                          \
  NSArray<MPPDetection *> *detectionsSubsetToCompare;                                              \
  XCTAssertEqual(objectDetectionResult.detections.count, expectedDetectionCount);                  \
  if (objectDetectionResult.detections.count > expectedObjectDetectionResult.detections.count) {   \
    detectionsSubsetToCompare = [objectDetectionResult.detections                                  \
        subarrayWithRange:NSMakeRange(0, expectedObjectDetectionResult.detections.count)];         \
  } else {                                                                                         \
    detectionsSubsetToCompare = objectDetectionResult.detections;                                  \
  }                                                                                                \
  \               
  AssertEqualDetectionArrays(detectionsSubsetToCompare, expectedObjectDetectionResult.detections); \
  XCTAssertEqual(objectDetectionResult.timestampMs, expectedObjectDetectionResult.timestampMs);

@interface MPPObjectDetectorTests : XCTestCase
@end

@implementation MPPObjectDetectorTests

#pragma mark Results

+ (MPPObjectDetectionResult *)expectedDetectionResultForCatsAndDogsImageWithTimestampMs:
    (NSInteger)timestampMs {
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

  return [[MPPObjectDetectionResult alloc] initWithDetections:detections timestampMs:timestampMs];
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
         equalsObjectDetectionResult:(MPPObjectDetectionResult *)expectedObjectDetectionResult {
  MPPObjectDetectionResult *objectDetectionResult = [objectDetector detectInImage:mppImage
                                                                            error:nil];
  AssertEqualObjectDetectionResults(
      objectDetectionResult, expectedObjectDetectionResult,
      maxResults > 0 ? maxResults : objectDetectionResult.detections.count);
}

- (void)assertResultsOfDetectInImageWithFileInfo:(NSDictionary *)fileInfo
                             usingObjectDetector:(MPPObjectDetector *)objectDetector
                                      maxResults:(NSInteger)maxResults

                     equalsObjectDetectionResult:
                         (MPPObjectDetectionResult *)expectedObjectDetectionResult {
  MPPImage *mppImage = [self imageWithFileInfo:fileInfo];

  [self assertResultsOfDetectInImage:mppImage
                 usingObjectDetector:objectDetector
                          maxResults:maxResults
         equalsObjectDetectionResult:expectedObjectDetectionResult];
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
                     equalsObjectDetectionResult:
                         [MPPObjectDetectorTests
                             expectedDetectionResultForCatsAndDogsImageWithTimestampMs:0]];
}

- (void)testDetectWithOptionsSucceeds {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  [self assertResultsOfDetectInImageWithFileInfo:kCatsAndDogsImage
                             usingObjectDetector:objectDetector
                                      maxResults:-1
                     equalsObjectDetectionResult:
                         [MPPObjectDetectorTests
                             expectedDetectionResultForCatsAndDogsImageWithTimestampMs:0]];
}

- (void)testDetectWithMaxResultsSucceeds {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  const NSInteger maxResults = 4;
  options.maxResults = maxResults;

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  [self assertResultsOfDetectInImageWithFileInfo:kCatsAndDogsImage
                             usingObjectDetector:objectDetector
                                      maxResults:maxResults
                     equalsObjectDetectionResult:
                         [MPPObjectDetectorTests
                             expectedDetectionResultForCatsAndDogsImageWithTimestampMs:0]];
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
  MPPObjectDetectionResult *expectedObjectDetectionResult =
      [[MPPObjectDetectionResult alloc] initWithDetections:detections timestampMs:0];

  [self assertResultsOfDetectInImageWithFileInfo:kCatsAndDogsImage
                             usingObjectDetector:objectDetector
                                      maxResults:-1
                     equalsObjectDetectionResult:expectedObjectDetectionResult];
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

  MPPObjectDetectionResult *expectedDetectionResult =
      [[MPPObjectDetectionResult alloc] initWithDetections:detections timestampMs:0];

  [self assertResultsOfDetectInImageWithFileInfo:kCatsAndDogsImage
                             usingObjectDetector:objectDetector
                                      maxResults:-1
                     equalsObjectDetectionResult:expectedDetectionResult];
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

  MPPObjectDetectionResult *expectedDetectionResult =
      [[MPPObjectDetectionResult alloc] initWithDetections:detections timestampMs:0];

  [self assertResultsOfDetectInImageWithFileInfo:kCatsAndDogsImage
                             usingObjectDetector:objectDetector
                                      maxResults:-1
                     equalsObjectDetectionResult:expectedDetectionResult];
}

- (void)testDetectWithOrientationSucceeds {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];
  options.maxResults = 1;

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  NSArray<MPPDetection *> *detections = @[
    [[MPPDetection alloc] initWithCategories:@[
      [[MPPCategory alloc] initWithIndex:-1
                                   score:0.750000f
                            categoryName:@"teddy bear"
                             displayName:nil],
    ]
                                 boundingBox:CGRectMake(0, 372, 416, 276)
                                   keypoints:nil],
  ];

  MPPObjectDetectionResult *expectedDetectionResult =
      [[MPPObjectDetectionResult alloc] initWithDetections:detections timestampMs:0];

  MPPImage *image = [self imageWithFileInfo:kCatsAndDogsRotatedImage
                                orientation:UIImageOrientationRight];

  [self assertResultsOfDetectInImage:image
                 usingObjectDetector:objectDetector
                          maxResults:1
         equalsObjectDetectionResult:expectedDetectionResult];
}

#pragma mark Running Mode Tests

- (void)testCreateObjectDetectorFailsWithResultListenerInNonLiveStreamMode {
  MPPRunningMode runningModesToTest[] = {MPPRunningModeImage, MPPRunningModeVideo};
  for (int i = 0; i < sizeof(runningModesToTest) / sizeof(runningModesToTest[0]); i++) {
    MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

    options.runningMode = runningModesToTest[i];
    options.completion =
        ^(MPPObjectDetectionResult *result, NSInteger timestampMs, NSError *error) {
        };

    [self
        assertCreateObjectDetectorWithOptions:options
                       failsWithExpectedError:
                           [NSError
                               errorWithDomain:kExpectedErrorDomain
                                          code:MPPTasksErrorCodeInvalidArgumentError
                                      userInfo:@{
                                        NSLocalizedDescriptionKey :
                                            @"The vision task is in image or video mode, a "
                                            @"user-defined result callback should not be provided."
                                      }]];
  }
}

- (void)testCreateObjectDetectorFailsWithMissingResultListenerInLiveStreamMode {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  options.runningMode = MPPRunningModeLiveStream;

  [self assertCreateObjectDetectorWithOptions:options
                       failsWithExpectedError:
                           [NSError errorWithDomain:kExpectedErrorDomain
                                               code:MPPTasksErrorCodeInvalidArgumentError
                                           userInfo:@{
                                             NSLocalizedDescriptionKey :
                                                 @"The vision task is in live stream mode, a "
                                                 @"user-defined result callback must be provided."
                                           }]];
}

- (void)testDetectFailsWithCallingWrongApiInImageMode {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kCatsAndDogsImage];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([objectDetector detectAsyncInImage:image
                                        timestampMs:0
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
  XCTAssertFalse([objectDetector detectInVideoFrame:image timestampMs:0 error:&videoApiCallError]);

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
  XCTAssertFalse([objectDetector detectAsyncInImage:image
                                        timestampMs:0
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
  XCTAssertFalse([objectDetector detectInImage:image error:&imageApiCallError]);

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
  options.completion = ^(MPPObjectDetectionResult *result, NSInteger timestampMs, NSError *error) {

  };

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kCatsAndDogsImage];

  NSError *imageApiCallError;
  XCTAssertFalse([objectDetector detectInImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Live Stream"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);

  NSError *videoApiCallError;
  XCTAssertFalse([objectDetector detectInVideoFrame:image timestampMs:0 error:&videoApiCallError]);

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
    MPPObjectDetectionResult *objectDetectionResult = [objectDetector detectInVideoFrame:image
                                                                             timestampMs:i
                                                                                   error:nil];
    AssertEqualObjectDetectionResults(
        objectDetectionResult,
        [MPPObjectDetectorTests expectedDetectionResultForCatsAndDogsImageWithTimestampMs:i],
        maxResults);
  }
}

- (void)testDetectWithOutOfOrderTimestampsAndLiveStreamModeFails {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  NSInteger maxResults = 4;
  options.maxResults = maxResults;

  options.runningMode = MPPRunningModeLiveStream;

  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"detectWithOutOfOrderTimestampsAndLiveStream"];
  expectation.expectedFulfillmentCount = 1;

  options.completion = ^(MPPObjectDetectionResult *result, NSInteger timestampMs, NSError *error) {
    AssertEqualObjectDetectionResults(
        result,
        [MPPObjectDetectorTests expectedDetectionResultForCatsAndDogsImageWithTimestampMs:1],
        maxResults);
    [expectation fulfill];
  };

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kCatsAndDogsImage];

  XCTAssertTrue([objectDetector detectAsyncInImage:image timestampMs:1 error:nil]);

  NSError *error;
  XCTAssertFalse([objectDetector detectAsyncInImage:image timestampMs:0 error:&error]);

  NSError *expectedError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey :
                            @"INVALID_ARGUMENT: Input timestamp must be monotonically increasing."
                      }];
  AssertEqualErrors(error, expectedError);
  [self waitForExpectations:@[ expectation ] timeout:0.1];
}

- (void)testDetectWithLiveStreamModeSucceeds {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  NSInteger maxResults = 4;
  options.maxResults = maxResults;

  options.runningMode = MPPRunningModeLiveStream;

  NSInteger iterationCount = 100;

  // Because of flow limiting, we cannot ensure that the callback will be
  // invoked `iterationCount` times.
  // An normal expectation will fail if expectation.fullfill() is not called
  // `expectation.expectedFulfillmentCount` times.
  // If `expectation.isInverted = true`, the test will only succeed if
  // expectation is not fullfilled for the specified `expectedFulfillmentCount`.
  // Since in our case we cannot predict how many times the expectation is
  // supposed to be fullfilled setting,
  // `expectation.expectedFulfillmentCount` = `iterationCount` + 1 and
  // `expectation.isInverted = true` ensures that test succeeds if
  // expectation is fullfilled <= `iterationCount` times.
  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"detectWithOutOfOrderTimestampsAndLiveStream"];
  expectation.expectedFulfillmentCount = iterationCount + 1;
  expectation.inverted = YES;

  options.completion = ^(MPPObjectDetectionResult *result, NSInteger timestampMs, NSError *error) {
    AssertEqualObjectDetectionResults(
        result,
        [MPPObjectDetectorTests
            expectedDetectionResultForCatsAndDogsImageWithTimestampMs:timestampMs],
        maxResults);
    [expectation fulfill];
  };

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  // TODO: Mimic initialization from CMSampleBuffer as live stream mode is most likely to be used
  // with the iOS camera. AVCaptureVideoDataOutput sample buffer delegates provide frames of type
  // `CMSampleBuffer`.
  MPPImage *image = [self imageWithFileInfo:kCatsAndDogsImage];

  for (int i = 0; i < iterationCount; i++) {
    XCTAssertTrue([objectDetector detectAsyncInImage:image timestampMs:i error:nil]);
  }

  [self waitForExpectations:@[ expectation ] timeout:0.5];
}

@end
