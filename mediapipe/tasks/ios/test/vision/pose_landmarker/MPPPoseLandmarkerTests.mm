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
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/test/vision/pose_landmarker/utils/sources/MPPPoseLandmarkerResult+ProtobufHelpers.h"
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPMask+TestUtils.h"
#import "mediapipe/tasks/ios/vision/pose_landmarker/sources/MPPPoseLandmarker.h"

static NSString *const kPbFileExtension = @"pbtxt";

static MPPFileInfo *const kPoseLandmarkerBundleAssetFileInfo =
    [[MPPFileInfo alloc] initWithName:@"pose_landmarker" type:@"task"];

static MPPFileInfo *const kPoseImageFileInfo = [[MPPFileInfo alloc] initWithName:@"pose"
                                                                            type:@"jpg"];
static MPPFileInfo *const kNoPoseImageFileInfo = [[MPPFileInfo alloc] initWithName:@"burger"
                                                                              type:@"jpg"];

static MPPFileInfo *const kExpectedPoseLandmarksFileInfo =
    [[MPPFileInfo alloc] initWithName:@"pose_landmarks" type:kPbFileExtension];

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";
static const float kLandmarksErrorTolerance = 0.03f;
static const float kVisibilityTolerance = 0.9f;
static const float kPresenceTolerance = 0.9f;

static NSString *const kLiveStreamTestsDictPoseLandmarkerKey = @"pose_landmarker";
static NSString *const kLiveStreamTestsDictExpectationKey = @"expectation";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

#define AssertApproximatelyEqualLandmarks(landmark, expectedLandmark, poseIndex, landmarkIndex)   \
  XCTAssertEqualWithAccuracy(landmark.x, expectedLandmark.x, kLandmarksErrorTolerance,            \
                             @"pose index = %d landmark index j = %d", poseIndex, landmarkIndex); \
  XCTAssertEqualWithAccuracy(landmark.y, expectedLandmark.y, kLandmarksErrorTolerance,            \
                             @"pose index = %d landmark index j = %d", poseIndex, landmarkIndex);

@interface MPPPoseLandmarkerTests : XCTestCase <MPPPoseLandmarkerLiveStreamDelegate> {
  NSDictionary<NSString *, id> *_liveStreamSucceedsTestDict;
  NSDictionary<NSString *, id> *_outOfOrderTimestampTestDict;
}
@end

@implementation MPPPoseLandmarkerTests

#pragma mark General Tests

- (void)testDetectWithModelPathSucceeds {
  MPPPoseLandmarker *poseLandmarker =
      [[MPPPoseLandmarker alloc] initWithModelPath:kPoseLandmarkerBundleAssetFileInfo.path
                                             error:nil];
  XCTAssertNotNil(poseLandmarker);

  [self assertResultsOfDetectInImageWithFileInfo:kPoseImageFileInfo
                             usingPoseLandmarker:poseLandmarker
         approximatelyEqualsPoseLandmarkerResult:[MPPPoseLandmarkerTests
                                                     expectedPoseLandmarkerResult]];
}

- (void)testDetectWithOptionsSucceeds {
  MPPPoseLandmarkerOptions *options =
      [self poseLandmarkerOptionsWithModelFileInfo:kPoseLandmarkerBundleAssetFileInfo];
  MPPPoseLandmarker *poseLandmarker = [self createPoseLandmarkerWithOptionsSucceeds:options];

  [self assertResultsOfDetectInImageWithFileInfo:kPoseImageFileInfo
                             usingPoseLandmarker:poseLandmarker
         approximatelyEqualsPoseLandmarkerResult:[MPPPoseLandmarkerTests
                                                     expectedPoseLandmarkerResult]];
}

- (void)testDetectWithEmptyResultsSucceeds {
  MPPPoseLandmarkerOptions *options =
      [self poseLandmarkerOptionsWithModelFileInfo:kPoseLandmarkerBundleAssetFileInfo];
  MPPPoseLandmarker *poseLandmarker = [self createPoseLandmarkerWithOptionsSucceeds:options];

  [self
      assertResultsOfDetectInImageWithFileInfo:kNoPoseImageFileInfo
                           usingPoseLandmarker:poseLandmarker
       approximatelyEqualsPoseLandmarkerResult:[MPPPoseLandmarkerTests emptyPoseLandmarkerResult]];
}

- (void)testCreatePoseLandmarkerFailsWithDelegateInNonLiveStreamMode {
  MPPRunningMode runningModesToTest[] = {MPPRunningModeImage, MPPRunningModeVideo};
  for (int i = 0; i < sizeof(runningModesToTest) / sizeof(runningModesToTest[0]); i++) {
    MPPPoseLandmarkerOptions *options =
        [self poseLandmarkerOptionsWithModelFileInfo:kPoseLandmarkerBundleAssetFileInfo];

    options.runningMode = runningModesToTest[i];
    options.poseLandmarkerLiveStreamDelegate = self;

    [self
        assertCreatePoseLandmarkerWithOptions:options
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

#pragma mark Running Mode Tests

- (void)testCreatePoseLandmarkerFailsWithMissingDelegateInLiveStreamMode {
  MPPPoseLandmarkerOptions *options =
      [self poseLandmarkerOptionsWithModelFileInfo:kPoseLandmarkerBundleAssetFileInfo];

  options.runningMode = MPPRunningModeLiveStream;

  [self assertCreatePoseLandmarkerWithOptions:options
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
  MPPPoseLandmarkerOptions *options =
      [self poseLandmarkerOptionsWithModelFileInfo:kPoseLandmarkerBundleAssetFileInfo];

  MPPPoseLandmarker *poseLandmarker = [self createPoseLandmarkerWithOptionsSucceeds:options];

  MPPImage *image = [MPPImage imageWithFileInfo:kPoseImageFileInfo];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([poseLandmarker detectAsyncImage:image
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
  XCTAssertFalse([poseLandmarker detectVideoFrame:image
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
  MPPPoseLandmarkerOptions *options =
      [self poseLandmarkerOptionsWithModelFileInfo:kPoseLandmarkerBundleAssetFileInfo];
  options.runningMode = MPPRunningModeVideo;

  MPPPoseLandmarker *poseLandmarker = [self createPoseLandmarkerWithOptionsSucceeds:options];

  MPPImage *image = [MPPImage imageWithFileInfo:kPoseImageFileInfo];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([poseLandmarker detectAsyncImage:image
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
  XCTAssertFalse([poseLandmarker detectImage:image error:&imageApiCallError]);

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
  MPPPoseLandmarkerOptions *options =
      [self poseLandmarkerOptionsWithModelFileInfo:kPoseLandmarkerBundleAssetFileInfo];
  options.runningMode = MPPRunningModeLiveStream;
  options.poseLandmarkerLiveStreamDelegate = self;

  MPPPoseLandmarker *poseLandmarker = [self createPoseLandmarkerWithOptionsSucceeds:options];

  MPPImage *image = [MPPImage imageWithFileInfo:kPoseImageFileInfo];

  NSError *imageApiCallError;
  XCTAssertFalse([poseLandmarker detectImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Live Stream"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);

  NSError *videoApiCallError;
  XCTAssertFalse([poseLandmarker detectVideoFrame:image
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
  MPPPoseLandmarkerOptions *options =
      [self poseLandmarkerOptionsWithModelFileInfo:kPoseLandmarkerBundleAssetFileInfo];
  options.runningMode = MPPRunningModeVideo;

  MPPPoseLandmarker *poseLandmarker = [self createPoseLandmarkerWithOptionsSucceeds:options];

  MPPImage *image = [MPPImage imageWithFileInfo:kPoseImageFileInfo];

  for (int i = 0; i < 3; i++) {
    MPPPoseLandmarkerResult *poseLandmarkerResult = [poseLandmarker detectVideoFrame:image
                                                             timestampInMilliseconds:i
                                                                               error:nil];
    [self assertPoseLandmarkerResult:poseLandmarkerResult
        isApproximatelyEqualToExpectedResult:[MPPPoseLandmarkerTests expectedPoseLandmarkerResult]];
  }
}

- (void)testDetectWithOutOfOrderTimestampsAndLiveStreamModeFails {
  MPPPoseLandmarkerOptions *options =
      [self poseLandmarkerOptionsWithModelFileInfo:kPoseLandmarkerBundleAssetFileInfo];
  options.runningMode = MPPRunningModeLiveStream;
  options.poseLandmarkerLiveStreamDelegate = self;

  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"detectWiththOutOfOrderTimestampsAndLiveStream"];

  expectation.expectedFulfillmentCount = 1;

  MPPPoseLandmarker *poseLandmarker = [self createPoseLandmarkerWithOptionsSucceeds:options];

  _outOfOrderTimestampTestDict = @{
    kLiveStreamTestsDictPoseLandmarkerKey : poseLandmarker,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  MPPImage *image = [MPPImage imageWithFileInfo:kPoseImageFileInfo];

  XCTAssertTrue([poseLandmarker detectAsyncImage:image timestampInMilliseconds:1 error:nil]);

  NSError *error;
  XCTAssertFalse([poseLandmarker detectAsyncImage:image timestampInMilliseconds:0 error:&error]);

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
  MPPPoseLandmarkerOptions *options =
      [self poseLandmarkerOptionsWithModelFileInfo:kPoseLandmarkerBundleAssetFileInfo];
  options.runningMode = MPPRunningModeLiveStream;
  options.poseLandmarkerLiveStreamDelegate = self;

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

  MPPPoseLandmarker *poseLandmarker = [self createPoseLandmarkerWithOptionsSucceeds:options];

  _liveStreamSucceedsTestDict = @{
    kLiveStreamTestsDictPoseLandmarkerKey : poseLandmarker,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  // TODO: Mimic initialization from CMSampleBuffer as live stream mode is most likely to be used
  // with the iOS camera. AVCaptureVideoDataOutput sample buffer delegates provide frames of type
  // `CMSampleBuffer`.
  MPPImage *image = [MPPImage imageWithFileInfo:kPoseImageFileInfo];

  for (int i = 0; i < iterationCount; i++) {
    XCTAssertTrue([poseLandmarker detectAsyncImage:image timestampInMilliseconds:i error:nil]);
  }

  NSTimeInterval timeout = 0.5f;
  [self waitForExpectations:@[ expectation ] timeout:timeout];
}

- (void)poseLandmarker:(MPPPoseLandmarker *)poseLandmarker
    didFinishDetectionWithResult:(MPPPoseLandmarkerResult *)poseLandmarkerResult
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(NSError *)error {
  [self assertPoseLandmarkerResult:poseLandmarkerResult
      isApproximatelyEqualToExpectedResult:[MPPPoseLandmarkerTests expectedPoseLandmarkerResult]];

  if (poseLandmarker == _outOfOrderTimestampTestDict[kLiveStreamTestsDictPoseLandmarkerKey]) {
    [_outOfOrderTimestampTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  } else if (poseLandmarker == _liveStreamSucceedsTestDict[kLiveStreamTestsDictPoseLandmarkerKey]) {
    [_liveStreamSucceedsTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  }
}

#pragma mark Pose Landmarker Initializers

- (MPPPoseLandmarkerOptions *)poseLandmarkerOptionsWithModelFileInfo:(MPPFileInfo *)modelFileInfo {
  MPPPoseLandmarkerOptions *poseLandmarkerOptions = [[MPPPoseLandmarkerOptions alloc] init];
  poseLandmarkerOptions.baseOptions.modelAssetPath = modelFileInfo.path;

  return poseLandmarkerOptions;
}

- (MPPPoseLandmarker *)createPoseLandmarkerWithOptionsSucceeds:
    (MPPPoseLandmarkerOptions *)poseLandmarkerOptions {
  NSError *error;
  MPPPoseLandmarker *poseLandmarker =
      [[MPPPoseLandmarker alloc] initWithOptions:poseLandmarkerOptions error:&error];
  XCTAssertNotNil(poseLandmarker);
  XCTAssertNil(error);

  return poseLandmarker;
}

- (void)assertCreatePoseLandmarkerWithOptions:(MPPPoseLandmarkerOptions *)poseLandmarkerOptions
                       failsWithExpectedError:(NSError *)expectedError {
  NSError *error = nil;
  MPPPoseLandmarker *poseLandmarker =
      [[MPPPoseLandmarker alloc] initWithOptions:poseLandmarkerOptions error:&error];

  XCTAssertNil(poseLandmarker);
  AssertEqualErrors(error, expectedError);
}

#pragma mark Results

+ (MPPPoseLandmarkerResult *)emptyPoseLandmarkerResult {
  return [[MPPPoseLandmarkerResult alloc] initWithLandmarks:@[]
                                             worldLandmarks:@[]
                                          segmentationMasks:@[]
                                    timestampInMilliseconds:0];
}

+ (MPPPoseLandmarkerResult *)expectedPoseLandmarkerResult {
  return [MPPPoseLandmarkerResult
      poseLandmarkerResultFromProtobufFileWithName:kExpectedPoseLandmarksFileInfo.path
                             shouldRemoveZPosition:YES];
}

- (void)assertResultsOfDetectInImageWithFileInfo:(MPPFileInfo *)fileInfo
                             usingPoseLandmarker:(MPPPoseLandmarker *)poseLandmarker
         approximatelyEqualsPoseLandmarkerResult:
             (MPPPoseLandmarkerResult *)expectedPoseLandmarkerResult {
  MPPPoseLandmarkerResult *poseLandmarkerResult = [self detectImageWithFileInfo:fileInfo
                                                            usingPoseLandmarker:poseLandmarker];
  [self assertPoseLandmarkerResult:poseLandmarkerResult
      isApproximatelyEqualToExpectedResult:expectedPoseLandmarkerResult];
}

- (MPPPoseLandmarkerResult *)detectImageWithFileInfo:(MPPFileInfo *)imageFileInfo
                                 usingPoseLandmarker:(MPPPoseLandmarker *)poseLandmarker {
  MPPImage *image = [MPPImage imageWithFileInfo:imageFileInfo];

  MPPPoseLandmarkerResult *poseLandmarkerResult = [poseLandmarker detectImage:image error:nil];
  XCTAssertNotNil(poseLandmarkerResult);

  return poseLandmarkerResult;
}

- (void)assertPoseLandmarkerResult:(MPPPoseLandmarkerResult *)poseLandmarkerResult
    isApproximatelyEqualToExpectedResult:(MPPPoseLandmarkerResult *)expectedPoseLandmarkerResult {
  // TODO: Add additional tests for auxiliary, world landmarks and segmentation masks.
  // Expects to have the same number of poses detected.
  [self assertMultiPoseLandmarks:poseLandmarkerResult.landmarks
      areApproximatelyEqualToExpectedMultiPoseLandmarks:expectedPoseLandmarkerResult.landmarks];

  [self assertLandmarksAreVisibleAndPresentInPoseLandmarkerResult:poseLandmarkerResult];
}

- (void)assertMultiPoseLandmarks:(NSArray<NSArray<MPPNormalizedLandmark *> *> *)multiPoseLandmarks
    areApproximatelyEqualToExpectedMultiPoseLandmarks:
        (NSArray<NSArray<MPPNormalizedLandmark *> *> *)expectedMultiPoseLandmarks {
  XCTAssertEqual(multiPoseLandmarks.count, expectedMultiPoseLandmarks.count);

  if (multiPoseLandmarks.count == 0) {
    return;
  }

  NSArray<MPPNormalizedLandmark *> *topPoseLandmarks = multiPoseLandmarks[0];
  NSArray<MPPNormalizedLandmark *> *expectedTopPoseLandmarks = expectedMultiPoseLandmarks[0];

  XCTAssertEqual(topPoseLandmarks.count, expectedTopPoseLandmarks.count);
  for (int i = 0; i < expectedTopPoseLandmarks.count; i++) {
    MPPNormalizedLandmark *landmark = topPoseLandmarks[i];
    XCTAssertNotNil(landmark);
    AssertApproximatelyEqualLandmarks(landmark, expectedTopPoseLandmarks[i], 0, i);
  }
}

- (void)assertLandmarksAreVisibleAndPresentInPoseLandmarkerResult:
    (MPPPoseLandmarkerResult *)poseLandmarkerResult {
  for (int i = 0; i < poseLandmarkerResult.landmarks.count; i++) {
    NSArray<MPPNormalizedLandmark *> *landmarks = poseLandmarkerResult.landmarks[i];
    for (int j = 0; j < landmarks.count; j++) {
      MPPNormalizedLandmark *landmark = landmarks[i];
      XCTAssertGreaterThanOrEqual(
          landmark.visibility.floatValue, kVisibilityTolerance,
          @"multi pose landmark index i = %d landmark index j = %d visibility %f", i, j,
          landmark.visibility.floatValue);
      XCTAssertGreaterThanOrEqual(
          landmark.presence.floatValue, kPresenceTolerance,
          @"multi pose landmark index i = %d landmark index j = %d presence %f", i, j,
          landmark.presence.floatValue);
    }
  }
}

@end
