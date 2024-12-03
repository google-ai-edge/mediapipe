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
#import <UIKit/UIKit.h>
#import <XCTest/XCTest.h>

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPDetection+Helpers.h"
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"
#import "mediapipe/tasks/ios/vision/face_detector/sources/MPPFaceDetector.h"
#import "mediapipe/tasks/ios/vision/face_detector/sources/MPPFaceDetectorResult.h"

static NSDictionary *const kPortraitImage =
    @{@"name" : @"portrait", @"type" : @"jpg", @"orientation" : @(UIImageOrientationUp)};
static NSDictionary *const kPortraitRotatedImage =
    @{@"name" : @"portrait_rotated", @"type" : @"jpg", @"orientation" : @(UIImageOrientationLeft)};
static NSDictionary *const kCatImage = @{@"name" : @"cat", @"type" : @"jpg"};
static NSString *const kShortRangeBlazeFaceModel = @"face_detection_short_range";
static NSArray<NSArray *> *const kPortraitExpectedKeypoints = @[
  @[ @0.44416f, @0.17643f ], @[ @0.55514f, @0.17731f ], @[ @0.50467f, @0.22657f ],
  @[ @0.50227f, @0.27199f ], @[ @0.36063f, @0.20143f ], @[ @0.60841f, @0.20409f ]
];
static NSArray<NSArray *> *const kPortraitRotatedExpectedKeypoints = @[
  @[ @0.82075f, @0.44679f ], @[ @0.81965f, @0.56261f ], @[ @0.76194f, @0.51719f ],
  @[ @0.71993f, @0.51719f ], @[ @0.80700f, @0.36298f ], @[ @0.80882f, @0.61204f ]
];
static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";
static NSString *const kLiveStreamTestsDictFaceDetectorKey = @"face_detector";
static NSString *const kLiveStreamTestsDictExpectationKey = @"expectation";

static const float kKeypointErrorThreshold = 1e-2;

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

@interface MPPFaceDetectorTests : XCTestCase <MPPFaceDetectorLiveStreamDelegate> {
  NSDictionary *liveStreamSucceedsTestDict;
  NSDictionary *outOfOrderTimestampTestDict;
}
@end

@implementation MPPFaceDetectorTests

#pragma mark General Tests

- (void)testCreateFaceDetectorWithMissingModelPathFails {
  NSString *modelPath = [MPPFaceDetectorTests filePathWithName:@"" extension:@""];

  NSError *error = nil;
  MPPFaceDetector *faceDetector = [[MPPFaceDetector alloc] initWithModelPath:modelPath
                                                                       error:&error];
  XCTAssertNil(faceDetector);

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

#pragma mark Image Mode Tests

- (void)testDetectWithImageModeAndPotraitSucceeds {
  NSString *modelPath = [MPPFaceDetectorTests filePathWithName:kShortRangeBlazeFaceModel
                                                     extension:@"tflite"];
  MPPFaceDetector *faceDetector = [[MPPFaceDetector alloc] initWithModelPath:modelPath error:nil];

  [self assertResultsOfDetectInImageWithFileInfo:kPortraitImage
                               usingFaceDetector:faceDetector
                       containsExpectedKeypoints:kPortraitExpectedKeypoints];
}

- (void)testDetectWithImageModeAndRotatedPotraitSucceeds {
  NSString *modelPath = [MPPFaceDetectorTests filePathWithName:kShortRangeBlazeFaceModel
                                                     extension:@"tflite"];
  MPPFaceDetector *faceDetector = [[MPPFaceDetector alloc] initWithModelPath:modelPath error:nil];
  XCTAssertNotNil(faceDetector);

  MPPImage *image = [self imageWithFileInfo:kPortraitRotatedImage];
  [self assertResultsOfDetectInImage:image
                   usingFaceDetector:faceDetector
           containsExpectedKeypoints:kPortraitRotatedExpectedKeypoints];
}

- (void)testDetectWithImageModeAndNoFaceSucceeds {
  NSString *modelPath = [MPPFaceDetectorTests filePathWithName:kShortRangeBlazeFaceModel
                                                     extension:@"tflite"];
  MPPFaceDetector *faceDetector = [[MPPFaceDetector alloc] initWithModelPath:modelPath error:nil];
  XCTAssertNotNil(faceDetector);

  NSError *error;
  MPPImage *mppImage = [self imageWithFileInfo:kCatImage];
  MPPFaceDetectorResult *faceDetectorResult = [faceDetector detectImage:mppImage error:&error];
  XCTAssertNil(error);
  XCTAssertNotNil(faceDetectorResult);
  XCTAssertEqual(faceDetectorResult.detections.count, 0);
}

#pragma mark Video Mode Tests

- (void)testDetectWithVideoModeAndPotraitSucceeds {
  MPPFaceDetectorOptions *options =
      [self faceDetectorOptionsWithModelName:kShortRangeBlazeFaceModel];
  options.runningMode = MPPRunningModeVideo;
  MPPFaceDetector *faceDetector = [self faceDetectorWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kPortraitImage];
  for (int i = 0; i < 3; i++) {
    MPPFaceDetectorResult *faceDetectorResult = [faceDetector detectVideoFrame:image
                                                       timestampInMilliseconds:i
                                                                         error:nil];
    [self assertFaceDetectorResult:faceDetectorResult
         containsExpectedKeypoints:kPortraitExpectedKeypoints];
  }
}

- (void)testDetectWithVideoModeAndRotatedPotraitSucceeds {
  MPPFaceDetectorOptions *options =
      [self faceDetectorOptionsWithModelName:kShortRangeBlazeFaceModel];
  options.runningMode = MPPRunningModeVideo;
  MPPFaceDetector *faceDetector = [self faceDetectorWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kPortraitRotatedImage];
  for (int i = 0; i < 3; i++) {
    MPPFaceDetectorResult *faceDetectorResult = [faceDetector detectVideoFrame:image
                                                       timestampInMilliseconds:i
                                                                         error:nil];
    [self assertFaceDetectorResult:faceDetectorResult
         containsExpectedKeypoints:kPortraitRotatedExpectedKeypoints];
  }
}

#pragma mark Live Stream Mode Tests

- (void)testDetectWithLiveStreamModeAndPotraitSucceeds {
  NSInteger iterationCount = 100;

  // Because of flow limiting, the callback might be invoked fewer than `iterationCount` times. An
  // normal expectation will fail if expectation.fulfill() is not called
  // `expectation.expectedFulfillmentCount` times. If `expectation.isInverted = true`, the test will
  // only succeed if expectation is not fulfilled for the specified `expectedFulfillmentCount`.
  // Since it is not possible to predict how many times the expectation is supposed to be
  // fulfilled, `expectation.expectedFulfillmentCount` = `iterationCount` + 1 and
  // `expectation.isInverted = true` ensures that test succeeds if expectation is fulfilled <=
  // `iterationCount` times.
  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"detectWithOutOfOrderTimestampsAndLiveStream"];
  expectation.expectedFulfillmentCount = iterationCount + 1;
  expectation.inverted = YES;

  MPPFaceDetectorOptions *options =
      [self faceDetectorOptionsWithModelName:kShortRangeBlazeFaceModel];
  options.runningMode = MPPRunningModeLiveStream;
  options.faceDetectorLiveStreamDelegate = self;

  MPPFaceDetector *faceDetector = [self faceDetectorWithOptionsSucceeds:options];
  MPPImage *image = [self imageWithFileInfo:kPortraitImage];

  liveStreamSucceedsTestDict = @{
    kLiveStreamTestsDictFaceDetectorKey : faceDetector,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  for (int i = 0; i < iterationCount; i++) {
    XCTAssertTrue([faceDetector detectAsyncImage:image timestampInMilliseconds:i error:nil]);
  }

  NSTimeInterval timeout = 0.5f;
  [self waitForExpectations:@[ expectation ] timeout:timeout];
}

- (void)testDetectWithOutOfOrderTimestampsAndLiveStreamModeFails {
  MPPFaceDetectorOptions *options =
      [self faceDetectorOptionsWithModelName:kShortRangeBlazeFaceModel];
  options.runningMode = MPPRunningModeLiveStream;
  options.faceDetectorLiveStreamDelegate = self;

  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"detectWithOutOfOrderTimestampsAndLiveStream"];
  expectation.expectedFulfillmentCount = 1;

  MPPFaceDetector *faceDetector = [self faceDetectorWithOptionsSucceeds:options];
  liveStreamSucceedsTestDict = @{
    kLiveStreamTestsDictFaceDetectorKey : faceDetector,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  MPPImage *image = [self imageWithFileInfo:kPortraitImage];
  XCTAssertTrue([faceDetector detectAsyncImage:image timestampInMilliseconds:1 error:nil]);

  NSError *error;
  XCTAssertFalse([faceDetector detectAsyncImage:image timestampInMilliseconds:0 error:&error]);

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

#pragma mark Running Mode Tests

- (void)testCreateFaceDetectorFailsWithDelegateInNonLiveStreamMode {
  MPPRunningMode runningModesToTest[] = {MPPRunningModeImage, MPPRunningModeVideo};
  for (int i = 0; i < sizeof(runningModesToTest) / sizeof(runningModesToTest[0]); i++) {
    MPPFaceDetectorOptions *options =
        [self faceDetectorOptionsWithModelName:kShortRangeBlazeFaceModel];

    options.runningMode = runningModesToTest[i];
    options.faceDetectorLiveStreamDelegate = self;

    [self assertCreateFaceDetectorWithOptions:options
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

- (void)testCreateFaceDetectorFailsWithMissingDelegateInLiveStreamMode {
  MPPFaceDetectorOptions *options =
      [self faceDetectorOptionsWithModelName:kShortRangeBlazeFaceModel];

  options.runningMode = MPPRunningModeLiveStream;

  [self assertCreateFaceDetectorWithOptions:options
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
  MPPFaceDetectorOptions *options =
      [self faceDetectorOptionsWithModelName:kShortRangeBlazeFaceModel];

  MPPFaceDetector *faceDetector = [self faceDetectorWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kPortraitImage];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([faceDetector detectAsyncImage:image
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
  XCTAssertFalse([faceDetector detectVideoFrame:image
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
  MPPFaceDetectorOptions *options =
      [self faceDetectorOptionsWithModelName:kShortRangeBlazeFaceModel];
  options.runningMode = MPPRunningModeVideo;

  MPPFaceDetector *faceDetector = [self faceDetectorWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kPortraitImage];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([faceDetector detectAsyncImage:image
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
  XCTAssertFalse([faceDetector detectImage:image error:&imageApiCallError]);

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
  MPPFaceDetectorOptions *options =
      [self faceDetectorOptionsWithModelName:kShortRangeBlazeFaceModel];

  options.runningMode = MPPRunningModeLiveStream;
  options.faceDetectorLiveStreamDelegate = self;

  MPPFaceDetector *faceDetector = [self faceDetectorWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kPortraitImage];

  NSError *imageApiCallError;
  XCTAssertFalse([faceDetector detectImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Live Stream"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);

  NSError *videoApiCallError;
  XCTAssertFalse([faceDetector detectVideoFrame:image
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

- (void)testDetectWithLiveStreamModeSucceeds {
  MPPFaceDetectorOptions *options =
      [self faceDetectorOptionsWithModelName:kShortRangeBlazeFaceModel];
  options.runningMode = MPPRunningModeLiveStream;
  options.faceDetectorLiveStreamDelegate = self;

  NSInteger iterationCount = 100;

  // Because of flow limiting, the callback might be invoked fewer than `iterationCount` times. An
  // normal expectation will fail if expectation.fulfill() is not called times. An normal
  // expectation will fail if expectation.fulfill() is not called
  // `expectation.expectedFulfillmentCount` times. If `expectation.isInverted = true`, the test will
  // only succeed if expectation is not fulfilled for the specified `expectedFulfillmentCount`.
  // Since it it not possible to determine how many times the expectation is supposed to be
  // fulfilled, `expectation.expectedFulfillmentCount` = `iterationCount` + 1 and
  // `expectation.isInverted = true` ensures that test succeeds if expectation is fulfilled <=
  // `iterationCount` times.
  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"detectWithOutOfOrderTimestampsAndLiveStream"];
  expectation.expectedFulfillmentCount = iterationCount + 1;
  expectation.inverted = YES;

  MPPFaceDetector *faceDetector = [self faceDetectorWithOptionsSucceeds:options];

  liveStreamSucceedsTestDict = @{
    kLiveStreamTestsDictFaceDetectorKey : faceDetector,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  MPPImage *image = [self imageWithFileInfo:kPortraitImage];
  for (int i = 0; i < iterationCount; i++) {
    XCTAssertTrue([faceDetector detectAsyncImage:image timestampInMilliseconds:i error:nil]);
  }

  NSTimeInterval timeout = 0.5f;
  [self waitForExpectations:@[ expectation ] timeout:timeout];
}

#pragma mark MPPFaceDetectorLiveStreamDelegate Methods
- (void)faceDetector:(MPPFaceDetector *)faceDetector
    didFinishDetectionWithResult:(MPPFaceDetectorResult *)faceDetectorResult
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(NSError *)error {
  [self assertFaceDetectorResult:faceDetectorResult
       containsExpectedKeypoints:kPortraitExpectedKeypoints];

  if (faceDetector == outOfOrderTimestampTestDict[kLiveStreamTestsDictFaceDetectorKey]) {
    [outOfOrderTimestampTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  } else if (faceDetector == liveStreamSucceedsTestDict[kLiveStreamTestsDictFaceDetectorKey]) {
    [liveStreamSucceedsTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  }
}

+ (NSString *)filePathWithName:(NSString *)fileName extension:(NSString *)extension {
  NSString *filePath =
      [[NSBundle bundleForClass:[MPPFaceDetectorTests class]] pathForResource:fileName
                                                                       ofType:extension];
  return filePath;
}

- (void)assertKeypoints:(NSArray<MPPNormalizedKeypoint *> *)keypoints
    areEqualToExpectedKeypoints:(NSArray<NSArray *> *)expectedKeypoint {
  XCTAssertEqual(keypoints.count, expectedKeypoint.count);
  for (int i = 0; i < keypoints.count; ++i) {
    XCTAssertEqualWithAccuracy(keypoints[i].location.x, [expectedKeypoint[i][0] floatValue],
                               kKeypointErrorThreshold, @"index i = %d", i);
    XCTAssertEqualWithAccuracy(keypoints[i].location.y, [expectedKeypoint[i][1] floatValue],
                               kKeypointErrorThreshold, @"index i = %d", i);
  }
}

- (void)assertDetections:(NSArray<MPPDetection *> *)detections
    containExpectedKeypoints:(NSArray<NSArray *> *)expectedKeypoints {
  XCTAssertEqual(detections.count, 1);
  MPPDetection *detection = detections[0];
  XCTAssertNotNil(detection);
  [self assertKeypoints:detections[0].keypoints areEqualToExpectedKeypoints:expectedKeypoints];
}

- (void)assertFaceDetectorResult:(MPPFaceDetectorResult *)faceDetectorResult
       containsExpectedKeypoints:(NSArray<NSArray *> *)expectedKeypoints {
  [self assertDetections:faceDetectorResult.detections containExpectedKeypoints:expectedKeypoints];
}

#pragma mark Face Detector Initializers

- (MPPFaceDetectorOptions *)faceDetectorOptionsWithModelName:(NSString *)modelName {
  NSString *modelPath = [MPPFaceDetectorTests filePathWithName:modelName extension:@"tflite"];
  MPPFaceDetectorOptions *faceDetectorOptions = [[MPPFaceDetectorOptions alloc] init];
  faceDetectorOptions.baseOptions.modelAssetPath = modelPath;

  return faceDetectorOptions;
}

- (void)assertCreateFaceDetectorWithOptions:(MPPFaceDetectorOptions *)faceDetectorOptions
                     failsWithExpectedError:(NSError *)expectedError {
  NSError *error = nil;
  MPPFaceDetector *faceDetector = [[MPPFaceDetector alloc] initWithOptions:faceDetectorOptions
                                                                     error:&error];
  XCTAssertNil(faceDetector);
  AssertEqualErrors(error, expectedError);
}

- (MPPFaceDetector *)faceDetectorWithOptionsSucceeds:(MPPFaceDetectorOptions *)faceDetectorOptions {
  MPPFaceDetector *faceDetector = [[MPPFaceDetector alloc] initWithOptions:faceDetectorOptions
                                                                     error:nil];
  XCTAssertNotNil(faceDetector);

  return faceDetector;
}

#pragma mark Assert Detection Results

- (MPPImage *)imageWithFileInfo:(NSDictionary *)fileInfo {
  UIImageOrientation orientation = (UIImageOrientation)[fileInfo[@"orientation"] intValue];
  MPPImage *image = [MPPImage imageFromBundleWithClass:[MPPFaceDetectorTests class]
                                              fileName:fileInfo[@"name"]
                                                ofType:fileInfo[@"type"]
                                           orientation:orientation];
  XCTAssertNotNil(image);
  return image;
}

- (void)assertResultsOfDetectInImage:(MPPImage *)mppImage
                   usingFaceDetector:(MPPFaceDetector *)faceDetector
           containsExpectedKeypoints:(NSArray<NSArray *> *)expectedKeypoints {
  NSError *error;
  MPPFaceDetectorResult *faceDetectorResult = [faceDetector detectImage:mppImage error:&error];
  XCTAssertNil(error);
  XCTAssertNotNil(faceDetectorResult);
  [self assertFaceDetectorResult:faceDetectorResult containsExpectedKeypoints:expectedKeypoints];
}

- (void)assertResultsOfDetectInImageWithFileInfo:(NSDictionary *)fileInfo
                               usingFaceDetector:(MPPFaceDetector *)faceDetector
                       containsExpectedKeypoints:(NSArray<NSArray *> *)expectedKeypoints {
  MPPImage *mppImage = [self imageWithFileInfo:fileInfo];

  [self assertResultsOfDetectInImage:mppImage
                   usingFaceDetector:faceDetector
           containsExpectedKeypoints:expectedKeypoints];
}

@end
