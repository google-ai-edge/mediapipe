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

@interface MPPHandLandmarkerTests : XCTestCase
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
  NSError* error;
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

- (MPPHandLandmarkerResult *)detectInImageWithFileInfo:(ResourceFileInfo *)imageFileInfo
                                   usingHandLandmarker:(MPPHandLandmarker *)handLandmarker {
  MPPImage *mppImage = [self imageWithFileInfo:imageFileInfo];
  MPPHandLandmarkerResult *handLandmarkerResult = [handLandmarker detectInImage:mppImage error:nil];
  XCTAssertNotNil(handLandmarkerResult);

  return handLandmarkerResult;
}

- (void)assertResultsOfDetectInImageWithFileInfo:(ResourceFileInfo *)fileInfo
                             usingHandLandmarker:(MPPHandLandmarker *)handLandmarker
         approximatelyEqualsHandLandmarkerResult:
             (MPPHandLandmarkerResult *)expectedHandLandmarkerResult {
  MPPHandLandmarkerResult *handLandmarkerResult = [self detectInImageWithFileInfo:fileInfo
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

  MPPHandLandmarkerResult *handLandmarkerResult = [self detectInImageWithFileInfo:kNoHandsImage
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

  MPPHandLandmarkerResult *handLandmarkerResult = [self detectInImageWithFileInfo:kTwoHandsImage
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

  MPPHandLandmarkerResult *handLandmarkerResult = [handLandmarker detectInImage:mppImage error:nil];

  [self assertHandLandmarkerResult:handLandmarkerResult
      isApproximatelyEqualToExpectedResult:[MPPHandLandmarkerTests
                                               pointingUpRotatedHandLandmarkerResult]];
}

@end
