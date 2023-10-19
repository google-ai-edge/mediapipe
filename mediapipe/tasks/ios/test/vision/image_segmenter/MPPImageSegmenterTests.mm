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

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPMask+TestUtils.h"
#import "mediapipe/tasks/ios/vision/image_segmenter/sources/MPPImageSegmenter.h"
#import "mediapipe/tasks/ios/vision/image_segmenter/sources/MPPImageSegmenterResult.h"

#include <iostream>
#include <vector>

static MPPFileInfo *const kCatImageFileInfo = [[MPPFileInfo alloc] initWithName:@"cat" type:@"jpg"];
static MPPFileInfo *const kCatGoldenImageFileInfo = [[MPPFileInfo alloc] initWithName:@"cat_mask"
                                                                                 type:@"jpg"];
static MPPFileInfo *const kSegmentationImageFileInfo =
    [[MPPFileInfo alloc] initWithName:@"segmentation_input_rotation0" type:@"jpg"];
static MPPFileInfo *const kSegmentationGoldenImageFileInfo =
    [[MPPFileInfo alloc] initWithName:@"segmentation_golden_rotation0" type:@"png"];

static MPPFileInfo *const kMozartImageFileInfo = [[MPPFileInfo alloc] initWithName:@"mozart_square"
                                                                              type:@"jpg"];
static MPPFileInfo *const kMozart128x128SegmentationGoldenImageFileInfo =
    [[MPPFileInfo alloc] initWithName:@"selfie_segm_128_128_3_expected_mask" type:@"jpg"];
static MPPFileInfo *const kMozart144x256SegmentationGoldenImageFileInfo =
    [[MPPFileInfo alloc] initWithName:@"selfie_segm_144_256_3_expected_mask" type:@"jpg"];

static MPPFileInfo *const kImageSegmenterModelFileInfo =
    [[MPPFileInfo alloc] initWithName:@"deeplabv3" type:@"tflite"];
static MPPFileInfo *const kSelfie128x128ModelFileInfo =
    [[MPPFileInfo alloc] initWithName:@"selfie_segm_128_128_3" type:@"tflite"];
static MPPFileInfo *const kSelfie144x256ModelFileInfo =
    [[MPPFileInfo alloc] initWithName:@"selfie_segm_144_256_3" type:@"tflite"];

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";
static NSString *const kLiveStreamTestsDictImageSegmenterKey = @"image_segmenter";
static NSString *const kLiveStreamTestsDictExpectationKey = @"expectation";

constexpr float kSimilarityThreshold = 0.96f;
constexpr NSInteger kMagnificationFactor = 10;
constexpr NSInteger kExpectedDeeplabV3ConfidenceMaskCount = 21;
constexpr NSInteger kExpected128x128SelfieSegmentationConfidenceMaskCount = 2;
constexpr NSInteger kExpected144x256SelfieSegmentationConfidenceMaskCount = 1;

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

namespace {
double sum(const std::vector<float> &mask) {
  double sum = 0.0;
  for (const float &maskElement : mask) {
    sum += maskElement;
  }
  return sum;
}

std::vector<float> multiply(const float *mask1, const float *mask2, size_t size) {
  std::vector<float> multipliedMask;
  multipliedMask.reserve(size);

  for (int i = 0; i < size; i++) {
    multipliedMask.push_back(mask1[i] * mask2[i]);
  }

  return multipliedMask;
}

double softIOU(const float *mask1, const float *mask2, size_t size) {
  std::vector<float> interSectionVector = multiply(mask1, mask2, size);
  double interSectionSum = sum(interSectionVector);

  std::vector<float> m1m1Vector = multiply(mask1, mask1, size);
  double m1m1 = sum(m1m1Vector);

  std::vector<float> m2m2Vector = multiply(mask2, mask2, size);
  double m2m2 = sum(m2m2Vector);

  double unionSum = m1m1 + m2m2 - interSectionSum;

  return unionSum > 0.0 ? interSectionSum / unionSum : 0.0;
}
}  // namespace

@interface MPPImageSegmenterTests : XCTestCase <MPPImageSegmenterLiveStreamDelegate> {
  NSDictionary<NSString *, id> *_liveStreamSucceedsTestDict;
  NSDictionary<NSString *, id> *_outOfOrderTimestampTestDict;
}

@end

@implementation MPPImageSegmenterTests

#pragma mark General Tests

- (void)setUp {
  // When expected and actual mask sizes are not equal, iterating through mask data results in a
  // segmentation fault. Setting this property to `NO`, prevents each test case from executing the
  // remaining flow after a failure. Since expected and actual mask sizes are compared before
  // iterating through them, this prevents any illegal memory access.
  self.continueAfterFailure = NO;
}

+ (NSString *)filePathWithName:(NSString *)fileName extension:(NSString *)extension {
  NSString *filePath =
      [[NSBundle bundleForClass:[MPPImageSegmenterTests class]] pathForResource:fileName
                                                                         ofType:extension];
  return filePath;
}

#pragma mark Image Mode Tests

- (void)testSegmentWithCategoryMaskSucceeds {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModelFileInfo];
  options.shouldOutputConfidenceMasks = NO;
  options.shouldOutputCategoryMask = YES;

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  [self assertResultsOfSegmentImageWithFileInfo:kSegmentationImageFileInfo
                                           usingImageSegmenter:imageSegmenter
      approximatelyEqualsExpectedCategoryMaskImageWithFileInfo:kSegmentationGoldenImageFileInfo
                                     shouldHaveConfidenceMasks:NO];
}

- (void)testSegmentWithConfidenceMaskSucceeds {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModelFileInfo];

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  [self assertResultsOfSegmentImageWithFileInfo:kCatImageFileInfo
                                             usingImageSegmenter:imageSegmenter
                                         hasConfidenceMasksCount:
                                             kExpectedDeeplabV3ConfidenceMaskCount
      approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:kCatGoldenImageFileInfo
                                                         atIndex:8
                                          shouldHaveCategoryMask:NO];
}

- (void)testSegmentWith128x128SegmentationSucceeds {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kSelfie128x128ModelFileInfo];

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  [self assertResultsOfSegmentImageWithFileInfo:kMozartImageFileInfo
                                             usingImageSegmenter:imageSegmenter
                                         hasConfidenceMasksCount:
                                             kExpected128x128SelfieSegmentationConfidenceMaskCount
      approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:
          kMozart128x128SegmentationGoldenImageFileInfo
                                                         atIndex:1
                                          shouldHaveCategoryMask:NO];
}

- (void)testSegmentWith144x256SegmentationSucceeds {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kSelfie144x256ModelFileInfo];

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  [self assertResultsOfSegmentImageWithFileInfo:kMozartImageFileInfo
                                             usingImageSegmenter:imageSegmenter
                                         hasConfidenceMasksCount:
                                             kExpected144x256SelfieSegmentationConfidenceMaskCount
      approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:
          kMozart144x256SegmentationGoldenImageFileInfo
                                                         atIndex:0
                                          shouldHaveCategoryMask:NO];
}

#pragma mark Running Mode Tests

- (void)testCreateImageSegmenterFailsWithDelegateInNonLiveStreamMode {
  MPPRunningMode runningModesToTest[] = {MPPRunningModeImage, MPPRunningModeVideo};
  for (int i = 0; i < sizeof(runningModesToTest) / sizeof(runningModesToTest[0]); i++) {
    MPPImageSegmenterOptions *options =
        [self imageSegmenterOptionsWithModelFileInfo:kSelfie128x128ModelFileInfo];

    options.runningMode = runningModesToTest[i];
    options.imageSegmenterLiveStreamDelegate = self;

    [self
        assertCreateImageSegmenterWithOptions:options
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

- (void)testCreateImageSegmenterFailsWithMissingDelegateInLiveStreamMode {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kSelfie128x128ModelFileInfo];

  options.runningMode = MPPRunningModeLiveStream;

  [self assertCreateImageSegmenterWithOptions:options
                       failsWithExpectedError:
                           [NSError
                               errorWithDomain:kExpectedErrorDomain
                                          code:MPPTasksErrorCodeInvalidArgumentError
                                      userInfo:@{
                                        NSLocalizedDescriptionKey :
                                            @"The vision task is in live stream mode. An object "
                                            @"must be set as the delegate of the task in its "
                                            @"options to ensure asynchronous delivery of results."
                                      }]];
}

- (void)testSegmentFailsWithCallingWrongApiInImageMode {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModelFileInfo];

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  MPPImage *image = [MPPImage imageWithFileInfo:kCatImageFileInfo];
  XCTAssertNotNil(image);

  NSError *liveStreamApiCallError;
  XCTAssertFalse([imageSegmenter segmentAsyncImage:image
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
  XCTAssertFalse([imageSegmenter segmentVideoFrame:image
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

- (void)testSegmentFailsWithCallingWrongApiInVideoMode {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModelFileInfo];
  options.runningMode = MPPRunningModeVideo;

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  MPPImage *image = [MPPImage imageWithFileInfo:kCatImageFileInfo];
  XCTAssertNotNil(image);

  NSError *liveStreamApiCallError;
  XCTAssertFalse([imageSegmenter segmentAsyncImage:image
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
  XCTAssertFalse([imageSegmenter segmentImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Video"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);
}

- (void)testSegmentFailsWithCallingWrongApiInLiveStreamMode {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModelFileInfo];
  options.runningMode = MPPRunningModeLiveStream;
  options.imageSegmenterLiveStreamDelegate = self;

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  MPPImage *image = [MPPImage imageWithFileInfo:kCatImageFileInfo];
  XCTAssertNotNil(image);

  NSError *imageApiCallError;
  XCTAssertFalse([imageSegmenter segmentImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Live Stream"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);

  NSError *videoApiCallError;
  XCTAssertFalse([imageSegmenter segmentVideoFrame:image
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

- (void)testSegmentWithVideoModeSucceeds {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModelFileInfo];
  options.runningMode = MPPRunningModeVideo;

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  MPPImage *image = [MPPImage imageWithFileInfo:kCatImageFileInfo];
  XCTAssertNotNil(image);

  for (int i = 0; i < 3; i++) {
    MPPImageSegmenterResult *result = [imageSegmenter segmentVideoFrame:image
                                                timestampInMilliseconds:i
                                                                  error:nil];
    [self assertImageSegmenterResult:result
                                           hasConfidenceMasksCount:
                                               kExpectedDeeplabV3ConfidenceMaskCount
        approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:kCatGoldenImageFileInfo
                                                           atIndex:8
                                            shouldHaveCategoryMask:NO];
  }
}

- (void)testSegmentWithOutOfOrderTimestampsAndLiveStreamModeFails {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModelFileInfo];
  options.runningMode = MPPRunningModeLiveStream;
  options.imageSegmenterLiveStreamDelegate = self;

  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"segmentWithOutOfOrderTimestampsAndLiveStream"];

  expectation.expectedFulfillmentCount = 1;

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  _outOfOrderTimestampTestDict = @{
    kLiveStreamTestsDictImageSegmenterKey : imageSegmenter,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  MPPImage *image = [MPPImage imageWithFileInfo:kCatImageFileInfo];
  XCTAssertNotNil(image);

  XCTAssertTrue([imageSegmenter segmentAsyncImage:image timestampInMilliseconds:1 error:nil]);

  NSError *error;
  XCTAssertFalse([imageSegmenter segmentAsyncImage:image timestampInMilliseconds:0 error:&error]);

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

- (void)testSegmentWithLiveStreamModeSucceeds {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModelFileInfo];
  options.runningMode = MPPRunningModeLiveStream;
  options.imageSegmenterLiveStreamDelegate = self;

  NSInteger iterationCount = 100;

  // Because of flow limiting, we cannot ensure that the callback will be invoked `iterationCount`
  // times. An normal expectation will fail if expectation.fulfill() is not called
  // `expectation.expectedFulfillmentCount` times. If `expectation.isInverted = true`, the test will
  // only succeed if expectation is not fulfilled for the specified `expectedFulfillmentCount`.
  // Since in our case we cannot predict how many times the expectation is supposed to be fulfilled
  // setting, `expectation.expectedFulfillmentCount` = `iterationCount` + 1 and
  // `expectation.isInverted = true` ensures that test succeeds ifexpectation is fulfilled <=
  // `iterationCount` times.
  XCTestExpectation *expectation =
      [[XCTestExpectation alloc] initWithDescription:@"segmentWithLiveStream"];

  expectation.expectedFulfillmentCount = iterationCount + 1;
  expectation.inverted = YES;

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  _outOfOrderTimestampTestDict = @{
    kLiveStreamTestsDictImageSegmenterKey : imageSegmenter,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  // TODO: Mimic initialization from CMSampleBuffer as live stream mode is most likely to be used
  // with the iOS camera. AVCaptureVideoDataOutput sample buffer delegates provide frames of type
  // `CMSampleBuffer`.
  MPPImage *image = [MPPImage imageWithFileInfo:kCatImageFileInfo];
  XCTAssertNotNil(image);

  for (int i = 0; i < iterationCount; i++) {
    XCTAssertTrue([imageSegmenter segmentAsyncImage:image timestampInMilliseconds:i error:nil]);
  }

  NSTimeInterval timeout = 0.5f;
  [self waitForExpectations:@[ expectation ] timeout:timeout];
}

- (void)imageSegmenter:(MPPImageSegmenter *)imageSegmenter
    didFinishSegmentationWithResult:(MPPImageSegmenterResult *)imageSegmenterResult
            timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                              error:(NSError *)error {
  [self assertImageSegmenterResult:imageSegmenterResult
                                         hasConfidenceMasksCount:
                                             kExpectedDeeplabV3ConfidenceMaskCount
      approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:kCatGoldenImageFileInfo
                                                         atIndex:8
                                          shouldHaveCategoryMask:NO];

  if (imageSegmenter == _outOfOrderTimestampTestDict[kLiveStreamTestsDictImageSegmenterKey]) {
    [_outOfOrderTimestampTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  } else if (imageSegmenter == _liveStreamSucceedsTestDict[kLiveStreamTestsDictImageSegmenterKey]) {
    [_liveStreamSucceedsTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  }
}

#pragma mark Mask No Copy Tests

- (void)testSegmentWithNoCopyConfidenceMasksAndImageModeSucceeds {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModelFileInfo];

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  MPPImage *image = [MPPImage imageWithFileInfo:kCatImageFileInfo];
  [imageSegmenter segmentImage:image withCompletionHandler:^(MPPImageSegmenterResult *result, NSError *error) {
      [self assertImageSegmenterResult:result
                                         hasConfidenceMasksCount:
                                             kExpectedDeeplabV3ConfidenceMaskCount
      approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:kCatGoldenImageFileInfo
                                                         atIndex:8
                                          shouldHaveCategoryMask:NO];
  }];
}

- (void)testSegmentWithNoCopyConfidenceMasksAndVideoModeSucceeds {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModelFileInfo];
  options.runningMode = MPPRunningModeVideo;

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  const NSInteger timestampInMilliseconds = 0;

  MPPImage *image = [MPPImage imageWithFileInfo:kCatImageFileInfo];
  [imageSegmenter segmentVideoFrame:image timestampInMilliseconds:timestampInMilliseconds withCompletionHandler:^(MPPImageSegmenterResult *result, NSError *error) {
      [self assertImageSegmenterResult:result
                                         hasConfidenceMasksCount:
                                             kExpectedDeeplabV3ConfidenceMaskCount
      approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:kCatGoldenImageFileInfo
                                                         atIndex:8
                                          shouldHaveCategoryMask:NO];
  }];
}

#pragma mark GetLabelsTest

- (void)testGetLabelsSucceeds {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModelFileInfo];

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  NSArray<NSString *> *expectedLabels = @[
    @"background", @"aeroplane", @"bicycle",      @"bird",  @"boat",         @"bottle", @"bus",
    @"car",        @"cat",       @"chair",        @"cow",   @"dining table", @"dog",    @"horse",
    @"motorbike",  @"person",    @"potted plant", @"sheep", @"sofa",         @"train",  @"tv"
  ];

  XCTAssertEqualObjects(imageSegmenter.labels, expectedLabels);
}

#pragma mark - Image Segmenter Initializers

- (MPPImageSegmenterOptions *)imageSegmenterOptionsWithModelFileInfo:(MPPFileInfo *)fileInfo {
  MPPImageSegmenterOptions *options = [[MPPImageSegmenterOptions alloc] init];
  options.baseOptions.modelAssetPath = fileInfo.path;
  return options;
}

- (MPPImageSegmenter *)createImageSegmenterWithOptionsSucceeds:(MPPImageSegmenterOptions *)options {
  NSError *error;
  MPPImageSegmenter *imageSegmenter = [[MPPImageSegmenter alloc] initWithOptions:options
                                                                           error:&error];
  XCTAssertNotNil(imageSegmenter);
  XCTAssertNil(error);

  return imageSegmenter;
}

- (void)assertCreateImageSegmenterWithOptions:(MPPImageSegmenterOptions *)options
                       failsWithExpectedError:(NSError *)expectedError {
  NSError *error = nil;
  MPPImageSegmenter *imageSegmenter = [[MPPImageSegmenter alloc] initWithOptions:options
                                                                           error:&error];

  XCTAssertNil(imageSegmenter);
  AssertEqualErrors(error, expectedError);
}

#pragma mark Assert Segmenter Results
- (void)assertResultsOfSegmentImageWithFileInfo:(MPPFileInfo *)imageFileInfo
                                         usingImageSegmenter:(MPPImageSegmenter *)imageSegmenter
    approximatelyEqualsExpectedCategoryMaskImageWithFileInfo:
        (MPPFileInfo *)expectedCategoryMaskFileInfo
                                   shouldHaveConfidenceMasks:(BOOL)shouldHaveConfidenceMasks {
  MPPImageSegmenterResult *result = [self segmentImageWithFileInfo:imageFileInfo
                                               usingImageSegmenter:imageSegmenter];

  XCTAssertNotNil(result.categoryMask);

  if (shouldHaveConfidenceMasks) {
    XCTAssertNotNil(result.confidenceMasks);
  } else {
    XCTAssertNil(result.confidenceMasks);
  }

  [self assertCategoryMask:result.categoryMask
      approximatelyEqualsExpectedCategoryMaskImageWithFileInfo:expectedCategoryMaskFileInfo];
}

- (void)assertResultsOfSegmentImageWithFileInfo:(MPPFileInfo *)imageFileInfo
                                           usingImageSegmenter:(MPPImageSegmenter *)imageSegmenter
                                       hasConfidenceMasksCount:
                                           (NSUInteger)expectedConfidenceMasksCount
    approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:
        (MPPFileInfo *)expectedConfidenceMaskFileInfo
                                                       atIndex:(NSInteger)index
                                        shouldHaveCategoryMask:(BOOL)shouldHaveCategoryMask {
  MPPImageSegmenterResult *result = [self segmentImageWithFileInfo:imageFileInfo
                                               usingImageSegmenter:imageSegmenter];

  [self assertImageSegmenterResult:result
                                         hasConfidenceMasksCount:expectedConfidenceMasksCount
      approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:expectedConfidenceMaskFileInfo
                                                         atIndex:index
                                          shouldHaveCategoryMask:shouldHaveCategoryMask];
}

- (void)assertImageSegmenterResult:(MPPImageSegmenterResult *)result
                                       hasConfidenceMasksCount:
                                           (NSUInteger)expectedConfidenceMasksCount
    approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:
        (MPPFileInfo *)expectedConfidenceMaskFileInfo
                                                       atIndex:(NSInteger)index
                                        shouldHaveCategoryMask:(BOOL)shouldHaveCategoryMask {
  XCTAssertNotNil(result.confidenceMasks);

  XCTAssertEqual(result.confidenceMasks.count, expectedConfidenceMasksCount);

  if (shouldHaveCategoryMask) {
    XCTAssertNotNil(result.categoryMask);
  } else {
    XCTAssertNil(result.categoryMask);
  }

  XCTAssertLessThan(index, result.confidenceMasks.count);

  [self assertConfidenceMask:result.confidenceMasks[index]
      approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:expectedConfidenceMaskFileInfo];
}

- (MPPImageSegmenterResult *)segmentImageWithFileInfo:(MPPFileInfo *)fileInfo
                                  usingImageSegmenter:(MPPImageSegmenter *)imageSegmenter {
  MPPImage *image = [MPPImage imageWithFileInfo:fileInfo];
  XCTAssertNotNil(image);

  NSError *error;
  MPPImageSegmenterResult *result = [imageSegmenter segmentImage:image error:&error];
  XCTAssertNil(error);
  XCTAssertNotNil(result);

  return result;
}

- (void)assertCategoryMask:(MPPMask *)categoryMask
    approximatelyEqualsExpectedCategoryMaskImageWithFileInfo:
        (MPPFileInfo *)expectedCategoryMaskImageFileInfo {
  MPPMask *expectedCategoryMask =
      [[MPPMask alloc] initWithImageFileInfo:expectedCategoryMaskImageFileInfo];

  XCTAssertEqual(categoryMask.width, expectedCategoryMask.width);
  XCTAssertEqual(categoryMask.height, expectedCategoryMask.height);

  size_t maskSize = categoryMask.width * categoryMask.height;

  const UInt8 *categoryMaskPixelData = categoryMask.uint8Data;
  const UInt8 *expectedCategoryMaskPixelData = expectedCategoryMask.uint8Data;

  NSInteger consistentPixels = 0;

  for (int i = 0; i < maskSize; i++) {
    consistentPixels +=
        categoryMaskPixelData[i] * kMagnificationFactor == expectedCategoryMaskPixelData[i] ? 1 : 0;
  }

  XCTAssertGreaterThan((float)consistentPixels / (float)maskSize, kSimilarityThreshold);
}

- (void)assertConfidenceMask:(MPPMask *)confidenceMask
    approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:
        (MPPFileInfo *)expectedConfidenceMaskImageFileInfo {
  MPPMask *expectedConfidenceMask =
      [[MPPMask alloc] initWithImageFileInfo:expectedConfidenceMaskImageFileInfo];

  XCTAssertEqual(confidenceMask.width, expectedConfidenceMask.width);
  XCTAssertEqual(confidenceMask.height, expectedConfidenceMask.height);

  size_t maskSize = confidenceMask.width * confidenceMask.height;

  XCTAssertGreaterThan(
      softIOU(confidenceMask.float32Data, expectedConfidenceMask.float32Data, maskSize),
      kSimilarityThreshold);
}

@end
