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

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/test/vision/holistic_landmarker/utils/sources/MPPHolisticLandmarkerResult+ProtobufHelpers.h"
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"
#import "mediapipe/tasks/ios/vision/holistic_landmarker/sources/MPPHolisticLandmarker.h"

static NSString *const kPbFileExtension = @"pbtxt";

static MPPFileInfo *const kHolisticLandmarkerBundleAssetFileInfo =
    [[MPPFileInfo alloc] initWithName:@"holistic_landmarker" type:@"task"];

static MPPFileInfo *const kHolisticImageFileInfo =
    [[MPPFileInfo alloc] initWithName:@"male_full_height_hands" type:@"jpg"];

static MPPFileInfo *const kCatImageFileInfo = [[MPPFileInfo alloc] initWithName:@"cat" type:@"jpg"];

static MPPFileInfo *const kExpectedHolisticLandmarksFileInfo =
    [[MPPFileInfo alloc] initWithName:@"male_full_height_hands_result_cpu" type:kPbFileExtension];

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";
static const float kLandmarksErrorTolerance = 0.03f;
static const float kFaceBlendshapesErrorTolerance = 0.13f;

static NSString *const kLiveStreamTestsDictHolisticLandmarkerKey = @"holistic_landmarker";
static NSString *const kLiveStreamTestsDictExpectationKey = @"expectation";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

#define AssertApproximatelyEqualLandmarks(landmark, expectedLandmark, landmarkTypeName,     \
                                          landmarkIndex)                                    \
  XCTAssertEqualWithAccuracy(landmark.x, expectedLandmark.x, kLandmarksErrorTolerance,      \
                             @"landmark type = %@ landmark index = %d", landmarkTypeName,   \
                             landmarkIndex);                                                \
  XCTAssertEqualWithAccuracy(landmark.y, expectedLandmark.y, kLandmarksErrorTolerance,      \
                             @"landmark type = %@ landmark index j = %d", landmarkTypeName, \
                             landmarkIndex);

#define AssertEqualCategories(category, expectedCategory, categoryIndex, errorTolerance)       \
  XCTAssertEqual(category.index, expectedCategory.index, @"index i = %d", categoryIndex);      \
  XCTAssertEqualWithAccuracy(category.score, expectedCategory.score, errorTolerance,           \
                             @"index i = %d", categoryIndex);                                  \
  XCTAssertEqualObjects(category.categoryName, expectedCategory.categoryName, @"index i = %d", \
                        categoryIndex);                                                        \
  XCTAssertEqualObjects(category.displayName, expectedCategory.displayName, @"index i = %d",   \
                        categoryIndex);

@interface MPPHolisticLandmarkerTests : XCTestCase <MPPHolisticLandmarkerLiveStreamDelegate> {
  NSDictionary<NSString *, id> *_liveStreamSucceedsTestDict;
  NSDictionary<NSString *, id> *_outOfOrderTimestampTestDict;
}
@end

@implementation MPPHolisticLandmarkerTests

#pragma mark General Tests

- (void)testDetectWithModelPathSucceeds {
  MPPHolisticLandmarker *holisticLandmarker =
      [[MPPHolisticLandmarker alloc] initWithModelPath:kHolisticLandmarkerBundleAssetFileInfo.path
                                                 error:nil];

  XCTAssertNotNil(holisticLandmarker);

  [self assertResultsOfDetectInImageWithFileInfo:kHolisticImageFileInfo
                          usingHolisticLandmarker:holisticLandmarker
      approximatelyEqualsHolisticLandmarkerResult:
          [MPPHolisticLandmarkerTests
              expectedHolisticLandmarkerResultWithFileInfo:kExpectedHolisticLandmarksFileInfo
                                        hasFaceBlendshapes:NO]];
}

- (void)testDetectWithOptionsSucceeds {
  MPPHolisticLandmarkerOptions *options =
      [self holisticLandmarkerOptionsWithModelFileInfo:kHolisticLandmarkerBundleAssetFileInfo];
  MPPHolisticLandmarker *holisticLandmarker =
      [self createHolisticLandmarkerWithOptionsSucceeds:options];

  [self assertResultsOfDetectInImageWithFileInfo:kHolisticImageFileInfo
                          usingHolisticLandmarker:holisticLandmarker
      approximatelyEqualsHolisticLandmarkerResult:
          [MPPHolisticLandmarkerTests
              expectedHolisticLandmarkerResultWithFileInfo:kExpectedHolisticLandmarksFileInfo
                                        hasFaceBlendshapes:NO]];
}

- (void)testDetectWithEmptyResultSucceeds {
  MPPHolisticLandmarkerOptions *options =
      [self holisticLandmarkerOptionsWithModelFileInfo:kHolisticLandmarkerBundleAssetFileInfo];
  MPPHolisticLandmarker *holisticLandmarker =
      [self createHolisticLandmarkerWithOptionsSucceeds:options];

  MPPHolisticLandmarkerResult *result = [self detectImageWithFileInfo:kCatImageFileInfo
                                              usingHolisticLandmarker:holisticLandmarker];

  XCTAssertEqual(result.faceLandmarks.count, 0);
  XCTAssertEqual(result.faceBlendshapes, nil);
  XCTAssertEqual(result.poseSegmentationMask, nil);
}

- (void)testDetectWithFaceBlendshapesSucceeds {
  MPPHolisticLandmarkerOptions *options =
      [self holisticLandmarkerOptionsWithModelFileInfo:kHolisticLandmarkerBundleAssetFileInfo];
  options.outputFaceBlendshapes = YES;

  MPPHolisticLandmarker *holisticLandmarker =
      [self createHolisticLandmarkerWithOptionsSucceeds:options];

  [self assertResultsOfDetectInImageWithFileInfo:kHolisticImageFileInfo
                          usingHolisticLandmarker:holisticLandmarker
      approximatelyEqualsHolisticLandmarkerResult:
          [MPPHolisticLandmarkerTests
              expectedHolisticLandmarkerResultWithFileInfo:kExpectedHolisticLandmarksFileInfo
                                        hasFaceBlendshapes:YES]];
}

- (void)testDetectWithSegmentationMasksSucceeds {
  MPPHolisticLandmarkerOptions *options =
      [self holisticLandmarkerOptionsWithModelFileInfo:kHolisticLandmarkerBundleAssetFileInfo];
  options.outputPoseSegmentationMasks = YES;

  MPPHolisticLandmarker *holisticLandmarker =
      [self createHolisticLandmarkerWithOptionsSucceeds:options];

  [self assertResultsOfDetectInImageWithFileInfo:kHolisticImageFileInfo
                          usingHolisticLandmarker:holisticLandmarker
      approximatelyEqualsHolisticLandmarkerResult:
          [MPPHolisticLandmarkerTests
              expectedHolisticLandmarkerResultWithFileInfo:kExpectedHolisticLandmarksFileInfo
                                        hasFaceBlendshapes:NO]
                        isSegmentationMaskPresent:YES];
}

#pragma mark Running Mode Tests

- (void)testCreateHolisticLandmarkerFailsWithMissingDelegateInLiveStreamMode {
  MPPHolisticLandmarkerOptions *options =
      [self holisticLandmarkerOptionsWithModelFileInfo:kHolisticLandmarkerBundleAssetFileInfo];

  options.runningMode = MPPRunningModeLiveStream;

  [self assertCreateHolisticLandmarkerWithOptions:options
                           failsWithExpectedError:
                               [NSError
                                   errorWithDomain:kExpectedErrorDomain
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
  MPPHolisticLandmarkerOptions *options =
      [self holisticLandmarkerOptionsWithModelFileInfo:kHolisticLandmarkerBundleAssetFileInfo];

  MPPHolisticLandmarker *holisticLandmarker =
      [self createHolisticLandmarkerWithOptionsSucceeds:options];

  MPPImage *image = [MPPHolisticLandmarkerTests createImageWithFileInfo:kHolisticImageFileInfo];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([holisticLandmarker detectAsyncImage:image
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
  XCTAssertFalse([holisticLandmarker detectVideoFrame:image
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
  MPPHolisticLandmarkerOptions *options =
      [self holisticLandmarkerOptionsWithModelFileInfo:kHolisticLandmarkerBundleAssetFileInfo];
  options.runningMode = MPPRunningModeVideo;

  MPPHolisticLandmarker *holisticLandmarker =
      [self createHolisticLandmarkerWithOptionsSucceeds:options];

  MPPImage *image = [MPPHolisticLandmarkerTests createImageWithFileInfo:kHolisticImageFileInfo];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([holisticLandmarker detectAsyncImage:image
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
  XCTAssertFalse([holisticLandmarker detectImage:image error:&imageApiCallError]);

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
  MPPHolisticLandmarkerOptions *options =
      [self holisticLandmarkerOptionsWithModelFileInfo:kHolisticLandmarkerBundleAssetFileInfo];
  options.runningMode = MPPRunningModeLiveStream;
  options.holisticLandmarkerLiveStreamDelegate = self;

  MPPHolisticLandmarker *holisticLandmarker =
      [self createHolisticLandmarkerWithOptionsSucceeds:options];

  MPPImage *image = [MPPHolisticLandmarkerTests createImageWithFileInfo:kHolisticImageFileInfo];

  NSError *imageApiCallError;
  XCTAssertFalse([holisticLandmarker detectImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Live Stream"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);

  NSError *videoApiCallError;
  XCTAssertFalse([holisticLandmarker detectVideoFrame:image
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

- (void)testDetectWithVideoModeSucceeds {
  MPPHolisticLandmarkerOptions *options =
      [self holisticLandmarkerOptionsWithModelFileInfo:kHolisticLandmarkerBundleAssetFileInfo];
  options.runningMode = MPPRunningModeVideo;

  MPPHolisticLandmarker *holisticLandmarker =
      [self createHolisticLandmarkerWithOptionsSucceeds:options];

  MPPImage *image = [MPPHolisticLandmarkerTests createImageWithFileInfo:kHolisticImageFileInfo];

  for (int i = 0; i < 3; i++) {
    MPPHolisticLandmarkerResult *holisticLandmarkerResult =
        [holisticLandmarker detectVideoFrame:image timestampInMilliseconds:i error:nil];
    [self assertHolisticLandmarkerResult:holisticLandmarkerResult
        isApproximatelyEqualToExpectedResult:
            [MPPHolisticLandmarkerTests
                expectedHolisticLandmarkerResultWithFileInfo:kExpectedHolisticLandmarksFileInfo
                                          hasFaceBlendshapes:NO]];
  }
}

- (void)testDetectWithOutOfOrderTimestampsAndLiveStreamModeFails {
  MPPHolisticLandmarkerOptions *options =
      [self holisticLandmarkerOptionsWithModelFileInfo:kHolisticLandmarkerBundleAssetFileInfo];
  options.runningMode = MPPRunningModeLiveStream;
  options.holisticLandmarkerLiveStreamDelegate = self;

  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"detectWiththOutOfOrderTimestampsAndLiveStream"];

  expectation.expectedFulfillmentCount = 1;

  MPPHolisticLandmarker *holisticLandmarker =
      [self createHolisticLandmarkerWithOptionsSucceeds:options];

  _outOfOrderTimestampTestDict = @{
    kLiveStreamTestsDictHolisticLandmarkerKey : holisticLandmarker,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  MPPImage *image = [MPPHolisticLandmarkerTests createImageWithFileInfo:kHolisticImageFileInfo];

  XCTAssertTrue([holisticLandmarker detectAsyncImage:image timestampInMilliseconds:1 error:nil]);

  NSError *error;
  XCTAssertFalse([holisticLandmarker detectAsyncImage:image
                              timestampInMilliseconds:0
                                                error:&error]);

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
  MPPHolisticLandmarkerOptions *options =
      [self holisticLandmarkerOptionsWithModelFileInfo:kHolisticLandmarkerBundleAssetFileInfo];
  options.runningMode = MPPRunningModeLiveStream;
  options.holisticLandmarkerLiveStreamDelegate = self;

  NSInteger iterationCount = 100;

  // Because of flow limiting, we cannot ensure that the callback will be invoked `iterationCount`
  // times. An normal expectation will fail if expectation.fulfill() is not called
  // `expectation.expectedFulfillmentCount` times. If `expectation.isInverted = true`, the test will
  // only succeed if expectation is not fulfilled for the specified `expectedFulfillmentCount`.
  // Since in our case we cannot predict how many times the expectation is supposed to be fulfilled
  // setting, `expectation.expectedFulfillmentCount` = `iterationCount` + 1 and
  // `expectation.isInverted = true` ensures that test succeeds if the expectation is fulfilled <=
  // `iterationCount` times.
  XCTestExpectation *expectation =
      [[XCTestExpectation alloc] initWithDescription:@"detectWithLiveStream"];

  expectation.expectedFulfillmentCount = iterationCount + 1;
  expectation.inverted = YES;

  MPPHolisticLandmarker *holisticLandmarker =
      [self createHolisticLandmarkerWithOptionsSucceeds:options];

  _liveStreamSucceedsTestDict = @{
    kLiveStreamTestsDictHolisticLandmarkerKey : holisticLandmarker,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  // TODO: Mimic initialization from CMSampleBuffer as live stream mode is most likely to be used
  // with the iOS camera. AVCaptureVideoDataOutput sample buffer delegates provide frames of type
  // `CMSampleBuffer`.
  MPPImage *image = [MPPHolisticLandmarkerTests createImageWithFileInfo:kHolisticImageFileInfo];

  for (int i = 0; i < iterationCount; i++) {
    XCTAssertTrue([holisticLandmarker detectAsyncImage:image timestampInMilliseconds:i error:nil]);
  }

  NSTimeInterval timeout = 0.5f;
  [self waitForExpectations:@[ expectation ] timeout:timeout];
}

- (void)holisticLandmarker:(MPPHolisticLandmarker *)holisticLandmarker
    didFinishDetectionWithResult:(MPPHolisticLandmarkerResult *)holisticLandmarkerResult
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(NSError *)error {
  [self assertHolisticLandmarkerResult:holisticLandmarkerResult
      isApproximatelyEqualToExpectedResult:
          [MPPHolisticLandmarkerTests
              expectedHolisticLandmarkerResultWithFileInfo:kExpectedHolisticLandmarksFileInfo
                                        hasFaceBlendshapes:NO]];

  if (holisticLandmarker ==
      _outOfOrderTimestampTestDict[kLiveStreamTestsDictHolisticLandmarkerKey]) {
    [_outOfOrderTimestampTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  } else if (holisticLandmarker ==
             _liveStreamSucceedsTestDict[kLiveStreamTestsDictHolisticLandmarkerKey]) {
    [_liveStreamSucceedsTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  }
}

#pragma mark Holistic Landmarker Initializers

- (MPPHolisticLandmarkerOptions *)holisticLandmarkerOptionsWithModelFileInfo:
    (MPPFileInfo *)modelFileInfo {
  MPPHolisticLandmarkerOptions *holisticLandmarkerOptions =
      [[MPPHolisticLandmarkerOptions alloc] init];
  holisticLandmarkerOptions.baseOptions.modelAssetPath = modelFileInfo.path;

  return holisticLandmarkerOptions;
}

- (MPPHolisticLandmarker *)createHolisticLandmarkerWithOptionsSucceeds:
    (MPPHolisticLandmarkerOptions *)holisticLandmarkerOptions {
  NSError *error;
  MPPHolisticLandmarker *holisticLandmarker =
      [[MPPHolisticLandmarker alloc] initWithOptions:holisticLandmarkerOptions error:&error];
  XCTAssertNotNil(holisticLandmarker);
  XCTAssertNil(error);

  return holisticLandmarker;
}

- (void)assertCreateHolisticLandmarkerWithOptions:
            (MPPHolisticLandmarkerOptions *)holisticLandmarkerOptions
                           failsWithExpectedError:(NSError *)expectedError {
  NSError *error = nil;
  MPPHolisticLandmarker *holisticLandmarker =
      [[MPPHolisticLandmarker alloc] initWithOptions:holisticLandmarkerOptions error:&error];

  XCTAssertNil(holisticLandmarker);
  AssertEqualErrors(error, expectedError);
}

#pragma mark Results

+ (MPPHolisticLandmarkerResult *)emptyHolisticLandmarkerResult {
  MPPHolisticLandmarkerResult *result =
      [[MPPHolisticLandmarkerResult alloc] initWithFaceLandmarks:@[]
                                                 faceBlendshapes:nil
                                                   poseLandmarks:@[]
                                              poseWorldLandmarks:@[]
                                            poseSegmentationMask:nil
                                               leftHandLandmarks:@[]
                                          leftHandWorldLandmarks:@[]
                                              rightHandLandmarks:@[]
                                         rightHandWorldLandmarks:@[]
                                         timestampInMilliseconds:0];

  return result;
}

+ (MPPHolisticLandmarkerResult *)
    expectedHolisticLandmarkerResultWithFileInfo:(MPPFileInfo *)fileInfo
                              hasFaceBlendshapes:(BOOL)hasFaceBlendshapes {
  MPPHolisticLandmarkerResult *result = [MPPHolisticLandmarkerResult
      holisticLandmarkerResultFromProtobufFileWithName:fileInfo.path
                                    hasFaceBlendshapes:hasFaceBlendshapes];

  return result;
}

- (void)assertResultsOfDetectInImageWithFileInfo:(MPPFileInfo *)fileInfo
                         usingHolisticLandmarker:(MPPHolisticLandmarker *)holisticLandmarker
     approximatelyEqualsHolisticLandmarkerResult:
         (MPPHolisticLandmarkerResult *)expectedHolisticLandmarkerResult {
  [self assertResultsOfDetectInImageWithFileInfo:fileInfo
                          usingHolisticLandmarker:holisticLandmarker
      approximatelyEqualsHolisticLandmarkerResult:expectedHolisticLandmarkerResult
                        isSegmentationMaskPresent:NO];
}
- (void)assertResultsOfDetectInImageWithFileInfo:(MPPFileInfo *)fileInfo
                         usingHolisticLandmarker:(MPPHolisticLandmarker *)holisticLandmarker
     approximatelyEqualsHolisticLandmarkerResult:
         (MPPHolisticLandmarkerResult *)expectedHolisticLandmarkerResult
                       isSegmentationMaskPresent:(BOOL)isSegmentationMaskPresent {
  MPPImage *image = [MPPHolisticLandmarkerTests createImageWithFileInfo:fileInfo];

  MPPHolisticLandmarkerResult *holisticLandmarkerResult = [self detectImage:image
                                                    usingHolisticLandmarker:holisticLandmarker];
  [self assertHolisticLandmarkerResult:holisticLandmarkerResult
      isApproximatelyEqualToExpectedResult:expectedHolisticLandmarkerResult];

  if (isSegmentationMaskPresent) {
    XCTAssertNotNil(holisticLandmarkerResult.poseSegmentationMask);
    XCTAssertEqual(holisticLandmarkerResult.poseSegmentationMask.width, image.width);
    XCTAssertEqual(holisticLandmarkerResult.poseSegmentationMask.height, image.height);
  }
}

- (MPPHolisticLandmarkerResult *)detectImageWithFileInfo:(MPPFileInfo *)imageFileInfo
                                 usingHolisticLandmarker:
                                     (MPPHolisticLandmarker *)holisticLandmarker {
  MPPImage *image = [MPPHolisticLandmarkerTests createImageWithFileInfo:imageFileInfo];
  MPPHolisticLandmarkerResult *holisticLandmarkerResult = [self detectImage:image
                                                    usingHolisticLandmarker:holisticLandmarker];

  return holisticLandmarkerResult;
}

- (MPPHolisticLandmarkerResult *)detectImage:(MPPImage *)image
                     usingHolisticLandmarker:(MPPHolisticLandmarker *)holisticLandmarker {
  NSError *error;
  MPPHolisticLandmarkerResult *holisticLandmarkerResult = [holisticLandmarker detectImage:image
                                                                                    error:&error];
  XCTAssertNotNil(holisticLandmarkerResult);
  XCTAssertNil(error);

  return holisticLandmarkerResult;
}

- (void)assertHolisticLandmarkerResult:(MPPHolisticLandmarkerResult *)holisticLandmarkerResult
    isApproximatelyEqualToExpectedResult:
        (MPPHolisticLandmarkerResult *)expectedHolisticLandmarkerResult {
  [self assertNormalizedLandmarks:holisticLandmarkerResult.faceLandmarks
                                    withLandmarkTypeName:@"face_landmarks"
      areApproximatelyEqualToExpectedNormalizedLandmarks:expectedHolisticLandmarkerResult
                                                             .faceLandmarks];

  [self assertNormalizedLandmarks:holisticLandmarkerResult.poseLandmarks
                                    withLandmarkTypeName:@"pose_landmarks"
      areApproximatelyEqualToExpectedNormalizedLandmarks:expectedHolisticLandmarkerResult
                                                             .poseLandmarks];

  [self assertNormalizedLandmarks:holisticLandmarkerResult.leftHandLandmarks
                                    withLandmarkTypeName:@"left_hand_landmarks"
      areApproximatelyEqualToExpectedNormalizedLandmarks:expectedHolisticLandmarkerResult
                                                             .leftHandLandmarks];

  [self assertNormalizedLandmarks:holisticLandmarkerResult.rightHandLandmarks
                                    withLandmarkTypeName:@"right_hand_landmarks"
      areApproximatelyEqualToExpectedNormalizedLandmarks:expectedHolisticLandmarkerResult
                                                             .rightHandLandmarks];

  [self assertFaceBlendshapes:holisticLandmarkerResult.faceBlendshapes
      areApproximatelyEqualToExpectedFaceBlendshapes:expectedHolisticLandmarkerResult
                                                         .faceBlendshapes];
}

- (void)assertNormalizedLandmarks:(NSArray<MPPNormalizedLandmark *> *)normalizedLandmarks
                                  withLandmarkTypeName:(NSString *)landmarkTypeName
    areApproximatelyEqualToExpectedNormalizedLandmarks:
        (NSArray<MPPNormalizedLandmark *> *)expectedNormalizedLandmarks {
  XCTAssertEqual(normalizedLandmarks.count, expectedNormalizedLandmarks.count);

  for (int i = 0; i < expectedNormalizedLandmarks.count; i++) {
    MPPNormalizedLandmark *landmark = normalizedLandmarks[i];

    XCTAssertNotNil(landmark);
    AssertApproximatelyEqualLandmarks(landmark, expectedNormalizedLandmarks[i], landmarkTypeName,
                                      i);
  }
}

- (void)assertFaceBlendshapes:(MPPClassifications *)faceBlendshapes
    areApproximatelyEqualToExpectedFaceBlendshapes:(MPPClassifications *)expectedFaceBlendshapes {
  XCTAssertEqual(faceBlendshapes.categories.count, expectedFaceBlendshapes.categories.count);

  for (int i = 0; i < expectedFaceBlendshapes.categories.count; i++) {
    MPPCategory *faceBlendshape = faceBlendshapes.categories[i];
    XCTAssertNotNil(faceBlendshape);
    AssertEqualCategories(faceBlendshape, expectedFaceBlendshapes.categories[i], i,
                          kFaceBlendshapesErrorTolerance);
  }
}

#pragma mark Image

+ (MPPImage *)createImageWithFileInfo:(MPPFileInfo *)fileInfo {
  MPPImage *image = [MPPImage imageWithFileInfo:fileInfo];
  XCTAssertNotNil(image);

  return image;
}
@end
