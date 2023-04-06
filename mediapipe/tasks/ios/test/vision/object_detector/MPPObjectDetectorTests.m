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
static NSDictionary *const kCatsAndDogsRotatedImage = @{@"name" : @"cats_and_dogs_rotated", @"type" : @"jpg"};
static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";

#define PixelDifferenceTolerance 5.0f

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
    XCTAssertEqual(categories[i].index, expectedCategories[i].index, @"categories index i = %d", i); \
    XCTAssertEqualWithAccuracy(categories[i].score, expectedCategories[i].score, 1e-3,    \
                               @"categories index i = %d", i);                                       \
    XCTAssertEqualObjects(categories[i].categoryName, expectedCategories[i].categoryName, \
                          @"categories index i = %d", i);                                            \
    XCTAssertEqualObjects(categories[i].displayName, expectedCategories[i].displayName,   \
                          @"categories index i = %d", i);                                            \
  }

#define AssertApproximatelyEqualBoundingBoxes(boundingBox, expectedBoundingBox, idx)                            \
  XCTAssertEqualWithAccuracy(boundingBox.origin.x, expectedBoundingBox.origin.x, PixelDifferenceTolerance, @"index i = %d", idx);     \
  XCTAssertEqualWithAccuracy(boundingBox.origin.y, expectedBoundingBox.origin.y, PixelDifferenceTolerance,  @"index i = %d", idx);     \
  XCTAssertEqualWithAccuracy(boundingBox.size.width, expectedBoundingBox.size.width, PixelDifferenceTolerance,  @"index i = %d", idx);     \
  XCTAssertEqualWithAccuracy(boundingBox.size.height, expectedBoundingBox.size.height, PixelDifferenceTolerance,  @"index i = %d", idx);

#define AssertEqualDetections(detection, expectedDetection, idx)                            \
  XCTAssertNotNil(detection);                                              \
  XCTAssertNil(detection.keypoints, @"index i = %d", idx) \
  AssertEqualCategoryArrays(detection.categories, expectedDetection.categories) \
  AssertApproximatelyEqualBoundingBoxes(detection.boundingBox, expectedDetection.boundingBox, idx);

#define AssertEqualDetectionArrays(detections, expectedDetections)                         \
  XCTAssertEqual(detections.count, expectedDetections.count);                             \
  for (int i = 0; i < detections.count; i++) {   \
    AssertEqualDetections(detections[i], expectedDetections[i], i)                                          \
  }

#define AssertEqualObjectDetectionResults(objectDetectionResult, expectedObjectDetectionResult)                   \
  XCTAssertNotNil(objectDetectionResult);                                              
  // AssertEqualDetectionArrays(objectDetectionResult.detections, expectedObjectDetectionResult.detections) \
  // XCTAssertEqual(objectDetectionResult.timestampMs, expectedObjectDetectionResult.timestampMs)

@interface MPPObjectDetectorTests : XCTestCase
@end

@implementation MPPObjectDetectorTests

#pragma mark Results

+ (MPPObjectDetectionResult*)expectedDetectionResultForCatsAndDogsImageWithTimestampMs:(NSInteger)timestampMs {

  NSArray<MPPDetection *> * detections = @[
    [[MPPDetection alloc] initWithCategories:@[
       [[MPPCategory alloc] initWithIndex:-1
                                 score:0.69921875f
                          categoryName:@"cat"
                           displayName:nil],
    ] boundingBox:CGRectMake(608, 161, 381, 439) keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
       [[MPPCategory alloc] initWithIndex:-1
                                 score:0.64453125f
                          categoryName:@"cat"
                           displayName:nil],
    ] boundingBox:CGRectMake(60, 398, 386, 196) keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
       [[MPPCategory alloc] initWithIndex:-1
                                 score:0.51171875f
                          categoryName:@"cat"
                           displayName:nil],
    ] boundingBox:CGRectMake(256, 395, 173, 202) keypoints:nil],
    [[MPPDetection alloc] initWithCategories:@[
       [[MPPCategory alloc] initWithIndex:-1
                                 score:0.48828125f
                          categoryName:@"cat"
                           displayName:nil],
    ] boundingBox:CGRectMake(362, 191, 325, 419) keypoints:nil],
  ];

  MPPObjectDetectionResult *objectDetectionResult = [[MPPObjectDetectionResult alloc] initWithDetections:detections timestampMs:timestampMs];
  
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
             equalsObjectDetectionResult:(MPPObjectDetectionResult *)expectedObjectDetectionResult {
  MPPObjectDetectionResult *objectDetectionResult = [objectDetector detectInImage:mppImage
                                                                             error:nil];

  AssertEqualObjectDetectionResults(objectDetectionResult, [MPPObjectDetectorTests expectedDetectionResultForCatsAndDogsImageWithTimestampMs:0]);
}

- (void)assertResultsOfDetectInImageWithFileInfo:(NSDictionary *)fileInfo
                            usingObjectDetector:(MPPObjectDetector *)objectDetector
                         equalsObjectDetectionResult:(MPPObjectDetectionResult *)expectedObjectDetectionResult {
  MPPImage *mppImage = [self imageWithFileInfo:fileInfo];

  [self assertResultsOfDetectInImage:mppImage
                usingObjectDetector:objectDetector
             equalsObjectDetectionResult:expectedObjectDetectionResult];
}

#pragma mark General Tests

- (void)testDetectWithOptionsSucceeds {
  MPPObjectDetectorOptions *options = [self objectDetectorOptionsWithModelName:kModelName];

  const NSInteger maxResults = 4;
  options.maxResults = maxResults;

  MPPObjectDetector *objectDetector = [self objectDetectorWithOptionsSucceeds:options];

  [self
      assertResultsOfDetectInImageWithFileInfo:kCatsAndDogsImage
                          usingObjectDetector:objectDetector
                              equalsObjectDetectionResult:
                                  [MPPObjectDetectorTests
                                      expectedDetectionResultForCatsAndDogsImageWithTimestampMs:0]];
}

@end
