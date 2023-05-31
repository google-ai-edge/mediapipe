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

static NSDictionary *const kGestureRecognizerBundleAssetFile =
    @{@"name" : @"gesture_recognizer", @"type" : @"task"};

static NSDictionary *const kTwoHandsImage = @{@"name" : @"right_hands", @"type" : @"jpg"};
static NSDictionary *const kFistImage = @{@"name" : @"fist", @"type" : @"jpg"};
static NSDictionary *const kNoHandsImage = @{@"name" : @"cats_and_dogs", @"type" : @"jpg"};
static NSDictionary *const kThumbUpImage = @{@"name" : @"thumb_up", @"type" : @"jpg"};
static NSDictionary *const kPointingUpRotatedImage =
    @{@"name" : @"pointing_up_rotated", @"type" : @"jpg"};

static NSDictionary *const kExpectedFistLandmarksFile =
    @{@"name" : @"fist_landmarks", @"type" : @"pbtxt"};
static NSDictionary *const kExpectedThumbUpLandmarksFile =
    @{@"name" : @"thumb_up_landmarks", @"type" : @"pbtxt"};

static NSString *const kFistLabel = @"Closed_Fist";
static NSString *const kExpectedThumbUpLabel = @"Thumb_Up";
static NSString *const kExpectedPointingUpLabel = @"Pointing_Up";
static NSString *const kRockLabel = @"Rock";

static const NSInteger kGestureExpectedIndex = -1;

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";
static const float kLandmarksErrorTolerance = 0.03f;

#define AssertEqualErrors(error, expectedError)                                               \
  XCTAssertNotNil(error);                                                                     \
  XCTAssertEqualObjects(error.domain, expectedError.domain);                                  \
  XCTAssertEqual(error.code, expectedError.code);                                             \
  XCTAssertNotEqual(                                                                          \
      [error.localizedDescription rangeOfString:expectedError.localizedDescription].location, \
      NSNotFound)

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

@interface MPPGestureRecognizerTests : XCTestCase
@end

@implementation MPPGestureRecognizerTests

#pragma mark Results

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
      gestureRecognizerResultsFromTextEncodedProtobufFileWithName:filePath
                                                     gestureLabel:kExpectedThumbUpLabel
                                            shouldRemoveZPosition:YES];
}

+ (MPPGestureRecognizerResult *)fistGestureRecognizerResultWithLabel:(NSString *)gestureLabel {
  NSString *filePath = [MPPGestureRecognizerTests filePathWithFileInfo:kExpectedFistLandmarksFile];

  return [MPPGestureRecognizerResult
      gestureRecognizerResultsFromTextEncodedProtobufFileWithName:filePath
                                                     gestureLabel:gestureLabel
                                            shouldRemoveZPosition:YES];
}

- (void)assertMultiHandLandmarks:(NSArray<NSArray<MPPNormalizedLandmark *> *> *)multiHandLandmarks
    isApproximatelyEqualToExpectedMultiHandLandmarks:
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
    isApproximatelyEqualToExpectedMultiHandWorldLandmarks:
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
    isApproximatelyEqualToExpectedMultiHandGestures:
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
      isApproximatelyEqualToExpectedMultiHandLandmarks:expectedGestureRecognizerResult.landmarks];
  [self assertMultiHandWorldLandmarks:gestureRecognizerResult.worldLandmarks
      isApproximatelyEqualToExpectedMultiHandWorldLandmarks:expectedGestureRecognizerResult
                                                                .worldLandmarks];
  [self assertMultiHandGestures:gestureRecognizerResult.gestures
      isApproximatelyEqualToExpectedMultiHandGestures:expectedGestureRecognizerResult.gestures];
}

#pragma mark File

+ (NSString *)filePathWithFileInfo:(NSDictionary *)fileInfo {
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
    (NSDictionary *)modelFileInfo {
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

#pragma mark Assert Gesture Recognizer Results

- (MPPImage *)imageWithFileInfo:(NSDictionary *)fileInfo {
  MPPImage *image = [MPPImage imageFromBundleWithClass:[MPPGestureRecognizerTests class]
                                              fileName:fileInfo[@"name"]
                                                ofType:fileInfo[@"type"]];
  XCTAssertNotNil(image);

  return image;
}

- (MPPImage *)imageWithFileInfo:(NSDictionary *)fileInfo
                    orientation:(UIImageOrientation)orientation {
  MPPImage *image = [MPPImage imageFromBundleWithClass:[MPPGestureRecognizerTests class]
                                              fileName:fileInfo[@"name"]
                                                ofType:fileInfo[@"type"]
                                           orientation:orientation];
  XCTAssertNotNil(image);

  return image;
}

- (MPPGestureRecognizerResult *)recognizeImageWithFileInfo:(NSDictionary *)imageFileInfo
                                    usingGestureRecognizer:
                                        (MPPGestureRecognizer *)gestureRecognizer {
  MPPImage *mppImage = [self imageWithFileInfo:imageFileInfo];
  MPPGestureRecognizerResult *gestureRecognizerResult = [gestureRecognizer recognizeImage:mppImage
                                                                                    error:nil];
  XCTAssertNotNil(gestureRecognizerResult);

  return gestureRecognizerResult;
}

- (void)assertResultsOfRecognizeImageWithFileInfo:(NSDictionary *)fileInfo
                           usingGestureRecognizer:(MPPGestureRecognizer *)gestureRecognizer
       approximatelyEqualsGestureRecognizerResult:
           (MPPGestureRecognizerResult *)expectedGestureRecognizerResult {
  MPPGestureRecognizerResult *gestureRecognizerResult =
      [self recognizeImageWithFileInfo:fileInfo usingGestureRecognizer:gestureRecognizer];
  [self assertGestureRecognizerResult:gestureRecognizerResult
      isApproximatelyEqualToExpectedResult:expectedGestureRecognizerResult];
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

@end
