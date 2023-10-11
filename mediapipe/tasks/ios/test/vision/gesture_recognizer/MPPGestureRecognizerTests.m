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
#import "mediapipe/tasks/ios/test/vision/gesture_recognizer/utils/sources/MPPGestureRecognizerResult+ProtobufHelpers.h"
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"
#import "mediapipe/tasks/ios/vision/gesture_recognizer/sources/MPPGestureRecognizer.h"

static NSString *const kPbFileExtension = @"pbtxt";

typedef NSDictionary<NSString *, NSString *> ResourceFileInfo;

static ResourceFileInfo *const kGestureRecognizerBundleAssetFile =
    @{@"name" : @"gesture_recognizer", @"type" : @"task"};

static ResourceFileInfo *const kTwoHandsImage = @{@"name" : @"right_hands", @"type" : @"jpg"};
static ResourceFileInfo *const kFistImage = @{@"name" : @"fist", @"type" : @"jpg"};
static ResourceFileInfo *const kNoHandsImage = @{@"name" : @"cats_and_dogs", @"type" : @"jpg"};
static ResourceFileInfo *const kThumbUpImage = @{@"name" : @"thumb_up", @"type" : @"jpg"};
static ResourceFileInfo *const kPointingUpRotatedImage =
    @{@"name" : @"pointing_up_rotated", @"type" : @"jpg"};

static ResourceFileInfo *const kExpectedFistLandmarksFile =
    @{@"name" : @"fist_landmarks", @"type" : kPbFileExtension};
static ResourceFileInfo *const kExpectedThumbUpLandmarksFile =
    @{@"name" : @"thumb_up_landmarks", @"type" : kPbFileExtension};

static NSString *const kFistLabel = @"Closed_Fist";
static NSString *const kExpectedThumbUpLabel = @"Thumb_Up";
static NSString *const kExpectedPointingUpLabel = @"Pointing_Up";
static NSString *const kRockLabel = @"Rock";

static const NSInteger kGestureExpectedIndex = -1;

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";
static const float kLandmarksErrorTolerance = 0.03f;

static NSString *const kLiveStreamTestsDictGestureRecognizerKey = @"gesture_recognizer";
static NSString *const kLiveStreamTestsDictExpectationKey = @"expectation";

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

#define AssertEqualGestures(gesture, expectedGesture, handIndex, gestureIndex)                  \
  XCTAssertEqual(gesture.index, kGestureExpectedIndex, @"hand index = %d gesture index j = %d", \
                 handIndex, gestureIndex);                                                      \
  XCTAssertEqualObjects(gesture.categoryName, expectedGesture.categoryName,                     \
                        @"hand index = %d gesture index j = %d", handIndex, gestureIndex);

#define AssertApproximatelyEqualLandmarks(landmark, expectedLandmark, handIndex, landmarkIndex)   \
  XCTAssertEqualWithAccuracy(landmark.x, expectedLandmark.x, kLandmarksErrorTolerance,            \
                             @"hand index = %d landmark index j = %d", handIndex, landmarkIndex); \
  XCTAssertEqualWithAccuracy(landmark.y, expectedLandmark.y, kLandmarksErrorTolerance,            \
                             @"hand index = %d landmark index j = %d", handIndex, landmarkIndex);

#define AssertGestureRecognizerResultIsEmpty(gestureRecognizerResult) \
  XCTAssertTrue(gestureRecognizerResult.gestures.count == 0);         \
  XCTAssertTrue(gestureRecognizerResult.handedness.count == 0);       \
  XCTAssertTrue(gestureRecognizerResult.landmarks.count == 0);        \
  XCTAssertTrue(gestureRecognizerResult.worldLandmarks.count == 0);

@interface MPPGestureRecognizerTests : XCTestCase <MPPGestureRecognizerLiveStreamDelegate> {
  NSDictionary<NSString *, id> *_liveStreamSucceedsTestDict;
  NSDictionary<NSString *, id> *_outOfOrderTimestampTestDict;
}
@end

@implementation MPPGestureRecognizerTests

#pragma mark Expected Results

+ (MPPGestureRecognizerResult *)emptyGestureRecognizerResult {
  return [[MPPGestureRecognizerResult alloc] initWithGestures:@[]
                                                   handedness:@[]
                                                    landmarks:@[]
                                               worldLandmarks:@[]
                                      timestampInMilliseconds:0];
}

+ (MPPGestureRecognizerResult *)thumbUpGestureRecognizerResult {
  NSString *filePath =
      [MPPGestureRecognizerTests filePathWithFileInfo:kExpectedThumbUpLandmarksFile];

  return [MPPGestureRecognizerResult
      gestureRecognizerResultsFromProtobufFileWithName:filePath
                                          gestureLabel:kExpectedThumbUpLabel
                                 shouldRemoveZPosition:YES];
}

+ (MPPGestureRecognizerResult *)fistGestureRecognizerResultWithLabel:(NSString *)gestureLabel {
  NSString *filePath = [MPPGestureRecognizerTests filePathWithFileInfo:kExpectedFistLandmarksFile];

  return [MPPGestureRecognizerResult gestureRecognizerResultsFromProtobufFileWithName:filePath
                                                                         gestureLabel:gestureLabel
                                                                shouldRemoveZPosition:YES];
}

#pragma mark Assert Gesture Recognizer Results

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

- (void)assertMultiHandGestures:(NSArray<NSArray<MPPCategory *> *> *)multiHandGestures
    areApproximatelyEqualToExpectedMultiHandGestures:
        (NSArray<NSArray<MPPCategory *> *> *)expectedMultiHandGestures {
  XCTAssertEqual(multiHandGestures.count, expectedMultiHandGestures.count);
  if (multiHandGestures.count == 0) {
    return;
  }

  NSArray<MPPCategory *> *topHandGestures = multiHandGestures[0];
  NSArray<MPPCategory *> *expectedTopHandGestures = expectedMultiHandGestures[0];

  XCTAssertEqual(topHandGestures.count, expectedTopHandGestures.count);
  for (int i = 0; i < expectedTopHandGestures.count; i++) {
    MPPCategory *gesture = topHandGestures[i];
    XCTAssertNotNil(gesture);
    AssertEqualGestures(gesture, expectedTopHandGestures[i], 0, i);
  }
}

- (void)assertGestureRecognizerResult:(MPPGestureRecognizerResult *)gestureRecognizerResult
    isApproximatelyEqualToExpectedResult:
        (MPPGestureRecognizerResult *)expectedGestureRecognizerResult {
  [self assertMultiHandLandmarks:gestureRecognizerResult.landmarks
      areApproximatelyEqualToExpectedMultiHandLandmarks:expectedGestureRecognizerResult.landmarks];
  [self assertMultiHandWorldLandmarks:gestureRecognizerResult.worldLandmarks
      areApproximatelyEqualToExpectedMultiHandWorldLandmarks:expectedGestureRecognizerResult
                                                                 .worldLandmarks];
  [self assertMultiHandGestures:gestureRecognizerResult.gestures
      areApproximatelyEqualToExpectedMultiHandGestures:expectedGestureRecognizerResult.gestures];
}

- (void)assertResultsOfRecognizeImageWithFileInfo:(ResourceFileInfo *)fileInfo
                           usingGestureRecognizer:(MPPGestureRecognizer *)gestureRecognizer
       approximatelyEqualsGestureRecognizerResult:
           (MPPGestureRecognizerResult *)expectedGestureRecognizerResult {
  MPPGestureRecognizerResult *gestureRecognizerResult =
      [self recognizeImageWithFileInfo:fileInfo usingGestureRecognizer:gestureRecognizer];
  [self assertGestureRecognizerResult:gestureRecognizerResult
      isApproximatelyEqualToExpectedResult:expectedGestureRecognizerResult];
}

#pragma mark File

+ (NSString *)filePathWithFileInfo:(ResourceFileInfo *)fileInfo {
  NSString *filePath = [MPPGestureRecognizerTests filePathWithName:fileInfo[@"name"]
                                                         extension:fileInfo[@"type"]];
  return filePath;
}

+ (NSString *)filePathWithName:(NSString *)fileName extension:(NSString *)extension {
  NSString *filePath = [[NSBundle bundleForClass:self.class] pathForResource:fileName
                                                                      ofType:extension];
  return filePath;
}

#pragma mark Gesture Recognizer Initializers

- (MPPGestureRecognizerOptions *)gestureRecognizerOptionsWithModelFileInfo:
    (ResourceFileInfo *)modelFileInfo {
  NSString *modelPath = [MPPGestureRecognizerTests filePathWithFileInfo:modelFileInfo];
  MPPGestureRecognizerOptions *gestureRecognizerOptions =
      [[MPPGestureRecognizerOptions alloc] init];
  gestureRecognizerOptions.baseOptions.modelAssetPath = modelPath;

  return gestureRecognizerOptions;
}

- (MPPGestureRecognizer *)createGestureRecognizerWithOptionsSucceeds:
    (MPPGestureRecognizerOptions *)gestureRecognizerOptions {
  MPPGestureRecognizer *gestureRecognizer =
      [[MPPGestureRecognizer alloc] initWithOptions:gestureRecognizerOptions error:nil];
  XCTAssertNotNil(gestureRecognizer);

  return gestureRecognizer;
}

- (void)assertCreateGestureRecognizerWithOptions:
            (MPPGestureRecognizerOptions *)gestureRecognizerOptions
                          failsWithExpectedError:(NSError *)expectedError {
  NSError *error = nil;
  MPPGestureRecognizer *gestureRecognizer =
      [[MPPGestureRecognizer alloc] initWithOptions:gestureRecognizerOptions error:&error];

  XCTAssertNil(gestureRecognizer);
  AssertEqualErrors(error, expectedError);
}

#pragma mark Recognize Helpers

- (MPPImage *)imageWithFileInfo:(ResourceFileInfo *)fileInfo {
  MPPImage *image = [MPPImage imageFromBundleWithClass:[MPPGestureRecognizerTests class]
                                              fileName:fileInfo[@"name"]
                                                ofType:fileInfo[@"type"]];
  XCTAssertNotNil(image);

  return image;
}

- (MPPImage *)imageWithFileInfo:(ResourceFileInfo *)fileInfo
                    orientation:(UIImageOrientation)orientation {
  MPPImage *image = [MPPImage imageFromBundleWithClass:[MPPGestureRecognizerTests class]
                                              fileName:fileInfo[@"name"]
                                                ofType:fileInfo[@"type"]
                                           orientation:orientation];
  XCTAssertNotNil(image);

  return image;
}

- (MPPGestureRecognizerResult *)recognizeImageWithFileInfo:(ResourceFileInfo *)imageFileInfo
                                    usingGestureRecognizer:
                                        (MPPGestureRecognizer *)gestureRecognizer {
  MPPImage *mppImage = [self imageWithFileInfo:imageFileInfo];
  MPPGestureRecognizerResult *gestureRecognizerResult = [gestureRecognizer recognizeImage:mppImage
                                                                                    error:nil];
  XCTAssertNotNil(gestureRecognizerResult);

  return gestureRecognizerResult;
}

#pragma mark General Tests

- (void)testRecognizeWithModelPathSucceeds {
  NSString *modelPath =
      [MPPGestureRecognizerTests filePathWithFileInfo:kGestureRecognizerBundleAssetFile];
  MPPGestureRecognizer *gestureRecognizer =
      [[MPPGestureRecognizer alloc] initWithModelPath:modelPath error:nil];
  XCTAssertNotNil(gestureRecognizer);

  [self assertResultsOfRecognizeImageWithFileInfo:kThumbUpImage
                           usingGestureRecognizer:gestureRecognizer
       approximatelyEqualsGestureRecognizerResult:[MPPGestureRecognizerTests
                                                      thumbUpGestureRecognizerResult]];
}

- (void)testRecognizeWithEmptyResultsSucceeds {
  MPPGestureRecognizerOptions *gestureRecognizerOptions =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:gestureRecognizerOptions];

  MPPGestureRecognizerResult *gestureRecognizerResult =
      [self recognizeImageWithFileInfo:kNoHandsImage usingGestureRecognizer:gestureRecognizer];
  AssertGestureRecognizerResultIsEmpty(gestureRecognizerResult);
}

- (void)testRecognizeWithScoreThresholdSucceeds {
  MPPGestureRecognizerOptions *gestureRecognizerOptions =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];
  gestureRecognizerOptions.cannedGesturesClassifierOptions = [[MPPClassifierOptions alloc] init];
  gestureRecognizerOptions.cannedGesturesClassifierOptions.scoreThreshold = 0.5f;

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:gestureRecognizerOptions];

  MPPGestureRecognizerResult *gestureRecognizerResult =
      [self recognizeImageWithFileInfo:kThumbUpImage usingGestureRecognizer:gestureRecognizer];

  MPPGestureRecognizerResult *expectedGestureRecognizerResult =
      [MPPGestureRecognizerTests thumbUpGestureRecognizerResult];

  XCTAssertTrue(gestureRecognizerResult.gestures.count == 1);
  AssertEqualGestures(gestureRecognizerResult.gestures[0][0],
                      expectedGestureRecognizerResult.gestures[0][0], 0, 0);
}

- (void)testRecognizeWithNumHandsSucceeds {
  MPPGestureRecognizerOptions *gestureRecognizerOptions =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];

  const NSInteger numHands = 2;
  gestureRecognizerOptions.numHands = numHands;

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:gestureRecognizerOptions];

  MPPGestureRecognizerResult *gestureRecognizerResult =
      [self recognizeImageWithFileInfo:kTwoHandsImage usingGestureRecognizer:gestureRecognizer];

  XCTAssertTrue(gestureRecognizerResult.handedness.count == numHands);
}

- (void)testRecognizeWithRotationSucceeds {
  MPPGestureRecognizerOptions *gestureRecognizerOptions =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];

  gestureRecognizerOptions.numHands = 1;

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:gestureRecognizerOptions];
  MPPImage *mppImage = [self imageWithFileInfo:kPointingUpRotatedImage
                                   orientation:UIImageOrientationLeft];

  MPPGestureRecognizerResult *gestureRecognizerResult = [gestureRecognizer recognizeImage:mppImage
                                                                                    error:nil];

  XCTAssertNotNil(gestureRecognizerResult);

  XCTAssertEqual(gestureRecognizerResult.gestures.count, 1);
  XCTAssertEqualObjects(gestureRecognizerResult.gestures[0][0].categoryName,
                        kExpectedPointingUpLabel);
}

- (void)testRecognizeWithCannedGestureFistSucceeds {
  MPPGestureRecognizerOptions *gestureRecognizerOptions =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];

  gestureRecognizerOptions.numHands = 1;

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:gestureRecognizerOptions];

  [self assertResultsOfRecognizeImageWithFileInfo:kFistImage
                           usingGestureRecognizer:gestureRecognizer
       approximatelyEqualsGestureRecognizerResult:
           [MPPGestureRecognizerTests fistGestureRecognizerResultWithLabel:kFistLabel]];
}

- (void)testRecognizeWithAllowGestureFistSucceeds {
  MPPGestureRecognizerOptions *gestureRecognizerOptions =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];
  gestureRecognizerOptions.cannedGesturesClassifierOptions = [[MPPClassifierOptions alloc] init];
  gestureRecognizerOptions.cannedGesturesClassifierOptions.scoreThreshold = 0.5f;
  gestureRecognizerOptions.cannedGesturesClassifierOptions.categoryAllowlist = @[ kFistLabel ];

  gestureRecognizerOptions.numHands = 1;

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:gestureRecognizerOptions];

  [self assertResultsOfRecognizeImageWithFileInfo:kFistImage
                           usingGestureRecognizer:gestureRecognizer
       approximatelyEqualsGestureRecognizerResult:
           [MPPGestureRecognizerTests fistGestureRecognizerResultWithLabel:kFistLabel]];
}

- (void)testRecognizeWithDenyGestureFistSucceeds {
  MPPGestureRecognizerOptions *gestureRecognizerOptions =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];
  gestureRecognizerOptions.cannedGesturesClassifierOptions = [[MPPClassifierOptions alloc] init];
  gestureRecognizerOptions.cannedGesturesClassifierOptions.scoreThreshold = 0.5f;
  gestureRecognizerOptions.cannedGesturesClassifierOptions.categoryDenylist = @[ kFistLabel ];

  gestureRecognizerOptions.numHands = 1;

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:gestureRecognizerOptions];
  MPPGestureRecognizerResult *gestureRecognizerResult =
      [self recognizeImageWithFileInfo:kFistImage usingGestureRecognizer:gestureRecognizer];
  AssertGestureRecognizerResultIsEmpty(gestureRecognizerResult);
}

- (void)testRecognizeWithPreferAllowlistOverDenylistSucceeds {
  MPPGestureRecognizerOptions *gestureRecognizerOptions =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];
  gestureRecognizerOptions.cannedGesturesClassifierOptions = [[MPPClassifierOptions alloc] init];
  gestureRecognizerOptions.cannedGesturesClassifierOptions.scoreThreshold = 0.5f;
  gestureRecognizerOptions.cannedGesturesClassifierOptions.categoryAllowlist = @[ kFistLabel ];
  gestureRecognizerOptions.cannedGesturesClassifierOptions.categoryDenylist = @[ kFistLabel ];

  gestureRecognizerOptions.numHands = 1;

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:gestureRecognizerOptions];

  [self assertResultsOfRecognizeImageWithFileInfo:kFistImage
                           usingGestureRecognizer:gestureRecognizer
       approximatelyEqualsGestureRecognizerResult:
           [MPPGestureRecognizerTests fistGestureRecognizerResultWithLabel:kFistLabel]];
}

#pragma mark Running Mode Tests

- (void)testCreateGestureRecognizerFailsWithDelegateInNonLiveStreamMode {
  MPPRunningMode runningModesToTest[] = {MPPRunningModeImage, MPPRunningModeVideo};
  for (int i = 0; i < sizeof(runningModesToTest) / sizeof(runningModesToTest[0]); i++) {
    MPPGestureRecognizerOptions *options =
        [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];

    options.runningMode = runningModesToTest[i];
    options.gestureRecognizerLiveStreamDelegate = self;

    [self assertCreateGestureRecognizerWithOptions:options
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

- (void)testCreateGestureRecognizerFailsWithMissingDelegateInLiveStreamMode {
  MPPGestureRecognizerOptions *options =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];

  options.runningMode = MPPRunningModeLiveStream;

  [self
      assertCreateGestureRecognizerWithOptions:options
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

- (void)testRecognizeFailsWithCallingWrongApiInImageMode {
  MPPGestureRecognizerOptions *options =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kFistImage];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([gestureRecognizer recognizeAsyncImage:image
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
  XCTAssertFalse([gestureRecognizer recognizeVideoFrame:image
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

- (void)testRecognizeFailsWithCallingWrongApiInVideoMode {
  MPPGestureRecognizerOptions *options =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];
  options.runningMode = MPPRunningModeVideo;

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kFistImage];

  NSError *liveStreamApiCallError;
  XCTAssertFalse([gestureRecognizer recognizeAsyncImage:image
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
  XCTAssertFalse([gestureRecognizer recognizeImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Video"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);
}

- (void)testRecognizeFailsWithCallingWrongApiInLiveStreamMode {
  MPPGestureRecognizerOptions *options =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];
  options.runningMode = MPPRunningModeLiveStream;
  options.gestureRecognizerLiveStreamDelegate = self;

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kFistImage];

  NSError *imageApiCallError;
  XCTAssertFalse([gestureRecognizer recognizeImage:image error:&imageApiCallError]);

  NSError *expectedImageApiCallError =
      [NSError errorWithDomain:kExpectedErrorDomain
                          code:MPPTasksErrorCodeInvalidArgumentError
                      userInfo:@{
                        NSLocalizedDescriptionKey : @"The vision task is not initialized with "
                                                    @"image mode. Current Running Mode: Live Stream"
                      }];
  AssertEqualErrors(imageApiCallError, expectedImageApiCallError);

  NSError *videoApiCallError;
  XCTAssertFalse([gestureRecognizer recognizeVideoFrame:image
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

- (void)testRecognizeWithVideoModeSucceeds {
  MPPGestureRecognizerOptions *options =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];
  options.runningMode = MPPRunningModeVideo;

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:options];

  MPPImage *image = [self imageWithFileInfo:kThumbUpImage];

  for (int i = 0; i < 3; i++) {
    MPPGestureRecognizerResult *gestureRecognizerResult =
        [gestureRecognizer recognizeVideoFrame:image timestampInMilliseconds:i error:nil];
    [self assertGestureRecognizerResult:gestureRecognizerResult
        isApproximatelyEqualToExpectedResult:[MPPGestureRecognizerTests
                                                 thumbUpGestureRecognizerResult]];
  }
}

- (void)testRecognizeWithOutOfOrderTimestampsAndLiveStreamModeFails {
  MPPGestureRecognizerOptions *options =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];
  options.runningMode = MPPRunningModeLiveStream;
  options.gestureRecognizerLiveStreamDelegate = self;

  XCTestExpectation *expectation = [[XCTestExpectation alloc]
      initWithDescription:@"recognizeWithOutOfOrderTimestampsAndLiveStream"];

  expectation.expectedFulfillmentCount = 1;

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:options];

  _outOfOrderTimestampTestDict = @{
    kLiveStreamTestsDictGestureRecognizerKey : gestureRecognizer,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  MPPImage *image = [self imageWithFileInfo:kThumbUpImage];

  XCTAssertTrue([gestureRecognizer recognizeAsyncImage:image timestampInMilliseconds:1 error:nil]);

  NSError *error;
  XCTAssertFalse([gestureRecognizer recognizeAsyncImage:image
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

- (void)testRecognizeWithLiveStreamModeSucceeds {
  MPPGestureRecognizerOptions *options =
      [self gestureRecognizerOptionsWithModelFileInfo:kGestureRecognizerBundleAssetFile];
  options.runningMode = MPPRunningModeLiveStream;
  options.gestureRecognizerLiveStreamDelegate = self;

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
      [[XCTestExpectation alloc] initWithDescription:@"recognizeWithLiveStream"];

  expectation.expectedFulfillmentCount = iterationCount + 1;
  expectation.inverted = YES;

  MPPGestureRecognizer *gestureRecognizer =
      [self createGestureRecognizerWithOptionsSucceeds:options];

  _liveStreamSucceedsTestDict = @{
    kLiveStreamTestsDictGestureRecognizerKey : gestureRecognizer,
    kLiveStreamTestsDictExpectationKey : expectation
  };

  // TODO: Mimic initialization from CMSampleBuffer as live stream mode is most likely to be used
  // with the iOS camera. AVCaptureVideoDataOutput sample buffer delegates provide frames of type
  // `CMSampleBuffer`.
  MPPImage *image = [self imageWithFileInfo:kThumbUpImage];

  for (int i = 0; i < iterationCount; i++) {
    XCTAssertTrue([gestureRecognizer recognizeAsyncImage:image
                                 timestampInMilliseconds:i
                                                   error:nil]);
  }

  NSTimeInterval timeout = 0.5f;
  [self waitForExpectations:@[ expectation ] timeout:timeout];
}

- (void)gestureRecognizer:(MPPGestureRecognizer *)gestureRecognizer
    didFinishRecognitionWithResult:(MPPGestureRecognizerResult *)gestureRecognizerResult
           timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                             error:(NSError *)error {
  [self assertGestureRecognizerResult:gestureRecognizerResult
      isApproximatelyEqualToExpectedResult:[MPPGestureRecognizerTests
                                               thumbUpGestureRecognizerResult]];

  if (gestureRecognizer == _outOfOrderTimestampTestDict[kLiveStreamTestsDictGestureRecognizerKey]) {
    [_outOfOrderTimestampTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  } else if (gestureRecognizer ==
             _liveStreamSucceedsTestDict[kLiveStreamTestsDictGestureRecognizerKey]) {
    [_liveStreamSucceedsTestDict[kLiveStreamTestsDictExpectationKey] fulfill];
  }
}

@end
