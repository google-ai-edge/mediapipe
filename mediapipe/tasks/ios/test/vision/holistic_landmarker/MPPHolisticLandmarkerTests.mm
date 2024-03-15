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

@interface MPPHolisticLandmarkerTests : XCTestCase
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
              expectedHolisticLandmarkerResultWithFileInfo:kExpectedHolisticLandmarksFileInfo]];
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
              expectedHolisticLandmarkerResultWithFileInfo:kExpectedHolisticLandmarksFileInfo]];
}

#pragma mark Pose Landmarker Initializers

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

+ (MPPHolisticLandmarkerResult *)expectedHolisticLandmarkerResultWithFileInfo:
    (MPPFileInfo *)fileInfo {
  MPPHolisticLandmarkerResult *result =
      [MPPHolisticLandmarkerResult holisticLandmarkerResultFromProtobufFileWithName:fileInfo.path];

  return result;
}

- (void)assertResultsOfDetectInImageWithFileInfo:(MPPFileInfo *)fileInfo
                         usingHolisticLandmarker:(MPPHolisticLandmarker *)holisticLandmarker
     approximatelyEqualsHolisticLandmarkerResult:
         (MPPHolisticLandmarkerResult *)expectedHolisticLandmarkerResult {
  MPPHolisticLandmarkerResult *holisticLandmarkerResult =
      [self detectImageWithFileInfo:fileInfo usingHolisticLandmarker:holisticLandmarker];
  [self assertHolisticLandmarkerResult:holisticLandmarkerResult
      isApproximatelyEqualToExpectedResult:expectedHolisticLandmarkerResult];
}

- (MPPHolisticLandmarkerResult *)detectImageWithFileInfo:(MPPFileInfo *)imageFileInfo
                                 usingHolisticLandmarker:
                                     (MPPHolisticLandmarker *)holisticLandmarker {
  MPPImage *image = [MPPImage imageWithFileInfo:imageFileInfo];

  NSError *error;

  MPPHolisticLandmarkerResult *holisticLandmarkerResult = [holisticLandmarker detectImage:image
                                                                                    error:&error];
  XCTAssertNotNil(holisticLandmarkerResult);

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

@end
