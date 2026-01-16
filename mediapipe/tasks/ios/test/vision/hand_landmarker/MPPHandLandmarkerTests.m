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
#import "mediapipe/tasks/ios/test/vision/hand_landmarker/utils/sources/MPPHandLandmarkerResult+ProtobufHelpers.h"
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"
#import "mediapipe/tasks/ios/vision/hand_landmarker/sources/MPPHandLandmarker.h"

static NSString *const kPbFileExtension = @"pbtxt";

typedef NSDictionary<NSString *, NSString *> ResourceFileInfo;

static ResourceFileInfo *const kHandLandmarkerBundleAssetFile =
    @{@"name" : @"hand_landmarker", @"type" : @"task"};

static ResourceFileInfo *const kTwoHandsImage = @{@"name" : @"right_hands", @"type" : @"jpg"};
static ResourceFileInfo *const kNoHandsImage = @{@"name" : @"cats_and_dogs", @"type" : @"jpg"};
static ResourceFileInfo *const kThumbUpImage = @{@"name" : @"thumb_up", @"type" : @"jpg"};
static ResourceFileInfo *const kPointingUpRotatedImage =
    @{@"name" : @"pointing_up_rotated", @"type" : @"jpg"};

static ResourceFileInfo *const kExpectedThumbUpLandmarksFile =
    @{@"name" : @"thumb_up_landmarks", @"type" : kPbFileExtension};
static ResourceFileInfo *const kExpectedPointingUpRotatedLandmarksFile =
    @{@"name" : @"pointing_up_rotated_landmarks", @"type" : kPbFileExtension};

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";
static const float kLandmarksErrorTolerance = 0.03f;

static NSString *const kLiveStreamTestsDictHandLandmarkerKey = @"gesture_recognizer";
static NSString *const kLiveStreamTestsDictExpectationKey = @"expectation";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

#define AssertApproximatelyEqualLandmarks(landmark, expectedLandmark, handIndex, landmarkIndex)   \
  XCTAssertEqualWithAccuracy(landmark.x, expectedLandmark.x, kLandmarksErrorTolerance,            \
                             @"hand index = %d landmark index j = %d", handIndex, landmarkIndex); \
  XCTAssertEqualWithAccuracy(landmark.y, expectedLandmark.y, kLandmarksErrorTolerance,            \
                             @"hand index = %d landmark index j = %d", handIndex, landmarkIndex);

#define AssertHandLandmarkerResultIsEmpty(handLandmarkerResult) \
  XCTAssertTrue(handLandmarkerResult.handedness.count == 0);    \
  XCTAssertTrue(handLandmarkerResult.landmarks.count == 0);     \
  XCTAssertTrue(handLandmarkerResult.worldLandmarks.count == 0);

@interface MPPHandLandmarkerTests : XCTestCase <MPPHandLandmarkerLiveStreamDelegate> {
  NSDictionary<NSString *, id> *_liveStreamSucceedsTestDict;
  NSDictionary<NSString *, id> *_outOfOrderTimestampTestDict;
}
@end

@implementation MPPHandLandmarkerTests

#pragma mark Results

+ (MPPHandLandmarkerResult *)emptyHandLandmarkerResult {
  return [[MPPHandLandmarkerResult alloc] initWithLandmarks:@[]
                                             worldLandmarks:@[]
                                                 handedness:@[]

                                    timestampInMilliseconds:0];
}

+ (MPPHandLandmarkerResult *)thumbUpHandLandmarkerResult {
  NSString *filePath = [MPPHandLandmarkerTests filePathWithFileInfo:kExpectedThumbUpLandmarksFile];

  return [MPPHandLandmarkerResult handLandmarkerResultFromProtobufFileWithName:filePath
                                                         shouldRemoveZPosition:YES];
}

+ (MPPHandLandmarkerResult *)pointingUpRotatedHandLandmarkerResult {
  NSString *filePath =
      [MPPHandLandmarkerTests filePathWithFileInfo:kExpectedPointingUpRotatedLandmarksFile];

  return [MPPHandLandmarkerResult handLandmarkerResultFromProtobufFileWithName:filePath
                                                         shouldRemoveZPosition:YES];
}

- (void)assertMultiHandLandmarks:(NSArray<NSArray<MPPNormalizedLandmark *> *> *)multiHandLandmarks
    areApproximatelyEqualToExpectedMultiHandLandmarks:
        (NSArray<NSArray<MPPNormalizedLandmark *> *> *)expectedMultiHandLandmarks {
  XCTAssertEqual(multiHandLandmarks.count, expectedMultiHandLandmarks.count);
  if (multiHandLandmarks.count == 0) {
    return;
  }

  NSArray<MPPNormalizedLandmark *> *topHandLandmarks = multiHandLandmarks[0];
  NSArray<MPPNormalizedLandmark *> *expectedTopHandLandmarks = expectedMultiHandLandmarks[0];

  XCTAssertEqual(topHandLandmarks.count, expectedTopHandLandmarks.count);
  for (int i = 0; i < expectedTopHandLandmarks.count; i++) {
    MPPNormalizedLandmark *landmark = topHandLandmarks[i];
    XCTAssertNotNil(landmark);
    AssertApproximatelyEqualLandmarks(landmark, expectedTopHandLandmarks[i], 0, i);
  }
}

- (void)assertMultiHandWorldLandmarks:(NSArray<NSArray<MPPLandmark *> *> *)multiHandWorldLandmarks
    areApproximatelyEqualToExpectedMultiHandWorldLandmarks:
        (NSArray<NSArray<MPPLandmark *> *> *)expectedMultiHandWorldLandmarks {
  XCTAssertEqual(multiHandWorldLandmarks.count, expectedMultiHandWorldLandmarks.count);
  if (expectedMultiHandWorldLandmarks.count == 0) {
    return;
  }

  NSArray<MPPLandmark *> *topHandWorldLandmarks = multiHandWorldLandmarks[0];
  NSArray<MPPLandmark *> *expectedTopHandWorldLandmarks = expectedMultiHandWorldLandmarks[0];

  XCTAssertEqual(topHandWorldLandmarks.count, expectedTopHandWorldLandmarks.count);
  for (int i = 0; i < expectedTopHandWorldLandmarks.count; i++) {
    MPPLandmark *landmark = topHandWorldLandmarks[i];
    XCTAssertNotNil(landmark);
    AssertApproximatelyEqualLandmarks(landmark, expectedTopHandWorldLandmarks[i], 0, i);
  }
}

- (void)assertHandLandmarkerResult:(MPPHandLandmarkerResult *)handLandmarkerResult
    isApproximatelyEqualToExpectedResult:(MPPHandLandmarkerResult *)expectedHandLandmarkerResult {
  [self assertMultiHandLandmarks:handLandmarkerResult.landmarks
      areApproximatelyEqualToExpectedMultiHandLandmarks:expectedHandLandmarkerResult.landmarks];
  [self assertMultiHandWorldLandmarks:handLandmarkerResult.worldLandmarks
      areApproximatelyEqualToExpectedMultiHandWorldLandmarks:expectedHandLandmarkerResult
                                                                 .worldLandmarks];
}

#pragma mark File

+ (NSString *)filePathWithFileInfo:(ResourceFileInfo *)fileInfo {
  NSString *filePath = [MPPHandLandmarkerTests filePathWithName:fileInfo[@"name"]
                                                      extension:fileInfo[@"type"]];
  return filePath;
}

+ (NSString *)filePathWithName:(NSString *)fileName extension:(NSString *)extension {
  NSString *filePath = [[NSBundle bundleForClass:self.class] pathForResource:fileName
                                                                      ofType:extension];
  return filePath;
}

#pragma mark Hand Landmarker Initializers

- (MPPHandLandmarkerOptions *)handLandmarkerOptionsWithModelFileInfo:
    (ResourceFileInfo *)modelFileInfo {
  NSString *modelPath = [MPPHandLandmarkerTests filePathWithFileInfo:modelFileInfo];
  MPPHandLandmarkerOptions *handLandmarkerOptions = [[MPPHandLandmarkerOptions alloc] init];
  handLandmarkerOptions.baseOptions.modelAssetPath = modelPath;

  return handLandmarkerOptions;
}

- (MPPHandLandmarker *)createHandLandmarkerWithOptionsSucceeds:
    (MPPHandLandmarkerOptions *)handLandmarkerOptions {
  NSError *error;
  MPPHandLandmarker *handLandmarker =
      [[MPPHandLandmarker alloc] initWithOptions:handLandmarkerOptions error:&error];
  XCTAssertNotNil(handLandmarker);
  XCTAssertNil(error);

  return handLandmarker;
}

- (void)assertCreateHandLandmarkerWithOptions:(MPPHandLandmarkerOptions *)handLandmarkerOptions
                       failsWithExpectedError:(NSError *)expectedError {
  NSError *error = nil;
  MPPHandLandmarker *handLandmarker =
      [[MPPHandLandmarker alloc] initWithOptions:handLandmarkerOptions error:&error];

  XCTAssertNil(handLandmarker);
  AssertEqualErrors(error, expectedError);
}

#pragma mark Assert Hand Landmarker Results

- (MPPImage *)imageWithFileInfo:(ResourceFileInfo *)fileInfo {
  MPPImage *image = [MPPImage imageFromBundleWithClass:[MPPHandLandmarkerTests class]
                                              fileName:fileInfo[@"name"]
                                                ofType:fileInfo[@"type"]];
  XCTAssertNotNil(image);

  return image;
}

- (MPPImage *)imageWithFileInfo:(ResourceFileInfo *)fileInfo
                    orientation:(UIImageOrientation)orientation {
  MPPImage *image = [MPPImage imageFromBundleWithClass:[MPPHandLandmarkerTests class]
                                              fileName:fileInfo[@"name"]
                                                ofType:fileInfo[@"type"]
                                           orientation:orientation];
  XCTAssertNotNil(image);

  return image;
}

- (MPPHandLandmarkerResult *)detectImageWithFileInfo:(ResourceFileInfo *)imageFileInfo
                                 usingHandLandmarker:(MPPHandLandmarker *)handLandmarker {
  MPPImage *mppImage = [self imageWithFileInfo:imageFileInfo];
  MPPHandLandmarkerResult *handLandmarkerResult = [handLandmarker detectImage:mppImage error:nil];
  XCTAssertNotNil(handLandmarkerResult);

  return handLandmarkerResult;
}

- (void)assertResultsOfDetectInImageWithFileInfo:(ResourceFileInfo *)fileInfo
                             usingHandLandmarker:(MPPHandLandmarker *)handLandmarker
         approximatelyEqualsHandLandmarkerResult:
             (MPPHandLandmarkerResult *)expectedHandLandmarkerResult {
  MPPHandLandmarkerResult *handLandmarkerResult = [self detectImageWithFileInfo:fileInfo
                                                            usingHandLandmarker:handLandmarker];
  [self assertHandLandmarkerResult:handLandmarkerResult
      isApproximatelyEqualToExpectedResult:expectedHandLandmarkerResult];
}

#pragma mark General Tests

- (void)testDetectWithModelPathSucceeds {
  NSString *modelPath =
      [MPPHandLandmarkerTests filePathWithFileInfo:kHandLandmarkerBundleAssetFile];
  MPPHandLandmarker *handLandmarker = [[MPPHandLandmarker alloc] initWithModelPath:modelPath
                                                                             error:nil];
  XCTAssertNotNil(handLandmarker);

  [self assertResultsOfDetectInImageWithFileInfo:kThumbUpImage
                             usingHandLandmarker:handLandmarker
         approximatelyEqualsHandLandmarkerResult:[MPPHandLandmarkerTests
                                                     thumbUpHandLandmarkerResult]];
}

- (void)testDetectWithEmptyResultsSucceeds {
  MPPHandLandmarkerOptions *handLandmarkerOptions =
      [self handLandmarkerOptionsWithModelFileInfo:kHandLandmarkerBundleAssetFile];

  MPPHandLandmarker *handLandmarker =
      [self createHandLandmarkerWithOptionsSucceeds:handLandmarkerOptions];

  MPPHandLandmarkerResult *handLandmarkerResult = [self detectImageWithFileInfo:kNoHandsImage
                                                            usingHandLandmarker:handLandmarker];
  AssertHandLandmarkerResultIsEmpty(handLandmarkerResult);
}

- (void)testDetectWithNumHandsSucceeds {
  MPPHandLandmarkerOptions *handLandmarkerOptions =
      [self handLandmarkerOptionsWithModelFileInfo:kHandLandmarkerBundleAssetFile];

  const NSInteger numHands = 2;
  handLandmarkerOptions.numHands = numHands;

  MPPHandLandmarker *handLandmarker =
      [self createHandLandmarkerWithOptionsSucceeds:handLandmarkerOptions];

  MPPHandLandmarkerResult *handLandmarkerResult = [self detectImageWithFileInfo:kTwoHandsImage
                                                            usingHandLandmarker:handLandmarker];

  XCTAssertTrue(handLandmarkerResult.handedness.count == numHands);
}

- (void)testDetectWithRotationSucceeds {
  MPPHandLandmarkerOptions *handLandmarkerOptions =
      [self handLandmarkerOptionsWithModelFileInfo:kHandLandmarkerBundleAssetFile];

  MPPHandLandmarker *handLandmarker =
      [self createHandLandmarkerWithOptionsSucceeds:handLandmarkerOptions];

  MPPImage *mppImage = [self imageWithFileInfo:kPointingUpRotatedImage
                                   orientation:UIImageOrientationRight];

  MPPHandLandmarkerResult *handLandmarkerResult = [handLandmarker detectImage:mppImage error:nil];

  [self assertHandLandmarkerResult:handLandmarkerResult
      isApproximatelyEqualToExpectedResult:[MPPHandLandmarkerTests
                                               pointingUpRotatedHandLandmarkerResult]];
}

#pragma mark Running Mode Tests

- (void)testCreateHandLandmarkerFailsWithDelegateInNonLiveStreamMode {
  MPPRunningMode runningModesToTest[] = {MPPRunningModeImage, MPPRunningModeVideo};
  for (int i = 0; i < sizeof(runningModesToTest) / sizeof(runningModesToTest[0]); i++) {
    MPPHandLandmarkerOptions *options =
        [self handLandmarkerOptionsWithModelFileInfo:kHandLandmarkerBundleAssetFile];

    options.runningMode = runningModesToTest[i];
    options.handLandmarkerLiveStreamDelegate = self;

    [self
        assertCreateHandLandmarkerWithOptions:options
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

- (void)testCreateHandLandmarkerFailsWithMissingDelegateInLiveStreamMode {
  MPPHandLandmarkerOptions *options =
      [self handLandmarkerOptionsWithModelFileInfo:kHandLandmarkerBundleAssetFile];

  options.runningMode = MPPRunningModeLiveStream;

  [self assertCreateHandLandmarkerWithOptions:options
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
  MPPHandLandmarkerOptions *options =
      [self handLandmarkerOptionsWithModelFileInfo:kHandLandmarkerBundleAssetFile];

  MPPHandLandmarker *handLandmarker = [self createHandLandmarkerWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kThumbUpImage];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([handLandmarker detectAsyncImage:image
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
  XCTAssertFalse([handLandmarker detectVideoFrame:image
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
  MPPHandLandmarkerOptions *options =
      [self handLandmarkerOptionsWithModelFileInfo:kHandLandmarkerBundleAssetFile];
  options.runningMode = MPPRunningModeVideo;

  MPPHandLandmarker *handLandmarker = [self createHandLandmarkerWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kThumbUpImage];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([handLandmarker detectAsyncImage:image
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
  XCTAssertFalse([handLandmarker detectImage:image error:&imageApiCallError]);

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
  MPPHandLandmarkerOptions *options =
      [self handLandmarkerOptionsWithModelFileInfo:kHandLandmarkerBundleAssetFile];
  options.runningMode = MPPRunningModeLiveStream;
  options.handLandmarkerLiveStreamDelegate = self;

  MPPHandLandmarker *handLandmarker = [self createHandLandmarkerWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kThumbUpImage];

  NSError *imageApiCallError;
  XCTAssertFalse([handLandmarker detectImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Live Stream"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);

  NSError *videoApiCallError;
  XCTAssertFalse([handLandmarker detectVideoFrame:image
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
  MPPHandLandmarkerOptions *options =
      [self handLandmarkerOptionsWithModelFileInfo:kHandLandmarkerBundleAssetFile];
  options.runningMode = MPPRunningModeVideo;

  MPPHandLandmarker *handLandmarker = [self createHandLandmarkerWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kThumbUpImage];

  for (int i = 0; i < 3; i++) {
    MPPHandLandmarkerResult *handLandmarkerResult = [handLandmarker detectVideoFrame:image
                                                             timestampInMilliseconds:i
                                                                               error:nil];
    [self assertHandLandmarkerResult:handLandmarkerResult
        isApproximatelyEqualToExpectedResult:[MPPHandLandmarkerTests thumbUpHandLandmarkerResult]];
  }
}

- (void)testDetectWithOutOfOrderTimestampsAndLiveStreamModeFails {
  MPPHandLandmarkerOptions *options =
      [self handLandmarkerOptionsWithModelFileInfo:kHandLandmarkerBundleAssetFile];
  options.runningMode = MPPRunningModeLiveStream;
  options.handLandmarkerLiveStreamDelegate = self;

  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"detectWiththOutOfOrderTimestampsAndLiveStream"];

  expectation.expectedFulfillmentCount = 1;

  MPPHandLandmarker *handLandmarker = [self createHandLandmarkerWithOptionsSucceeds:options];

  _outOfOrderTimestampTestDict = @{
    kLiveStreamTestsDictHandLandmarkerKey : handLandmarker,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  MPPImage *image = [self imageWithFileInfo:kThumbUpImage];

  XCTAssertTrue([handLandmarker detectAsyncImage:image timestampInMilliseconds:1 error:nil]);

  NSError *error;
  XCTAssertFalse([handLandmarker detectAsyncImage:image timestampInMilliseconds:0 error:&error]);

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
  MPPHandLandmarkerOptions *options =
      [self handLandmarkerOptionsWithModelFileInfo:kHandLandmarkerBundleAssetFile];
  options.runningMode = MPPRunningModeLiveStream;
  options.handLandmarkerLiveStreamDelegate = self;

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

  MPPHandLandmarker *handLandmarker = [self createHandLandmarkerWithOptionsSucceeds:options];

  _liveStreamSucceedsTestDict = @{
    kLiveStreamTestsDictHandLandmarkerKey : handLandmarker,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  // TODO: Mimic initialization from CMSampleBuffer as live stream mode is most likely to be used
  // with the iOS camera. AVCaptureVideoDataOutput sample buffer delegates provide frames of type
  // `CMSampleBuffer`.
  MPPImage *image = [self imageWithFileInfo:kThumbUpImage];

  for (int i = 0; i < iterationCount; i++) {
    XCTAssertTrue([handLandmarker detectAsyncImage:image timestampInMilliseconds:i error:nil]);
  }

  NSTimeInterval timeout = 0.5f;
  [self waitForExpectations:@[ expectation ] timeout:timeout];
}

- (void)handLandmarker:(MPPHandLandmarker *)handLandmarker
    didFinishDetectionWithResult:(MPPHandLandmarkerResult *)handLandmarkerResult
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(NSError *)error {
  [self assertHandLandmarkerResult:handLandmarkerResult
      isApproximatelyEqualToExpectedResult:[MPPHandLandmarkerTests thumbUpHandLandmarkerResult]];

  if (handLandmarker == _outOfOrderTimestampTestDict[kLiveStreamTestsDictHandLandmarkerKey]) {
    [_outOfOrderTimestampTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  } else if (handLandmarker == _liveStreamSucceedsTestDict[kLiveStreamTestsDictHandLandmarkerKey]) {
    [_liveStreamSucceedsTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  }
}

@end
