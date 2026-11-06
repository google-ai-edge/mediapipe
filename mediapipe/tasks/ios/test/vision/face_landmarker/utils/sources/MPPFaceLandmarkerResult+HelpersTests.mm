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

#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker_result.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/sources/MPPFaceLandmarkerResult.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/utils/sources/MPPFaceLandmarkerResult+Helpers.h"

@interface MPPLandmarkerResultHelpersTests : XCTestCase {
}
@end

@implementation MPPLandmarkerResultHelpersTests

- (void)testCreatesResultFromCResult {
  MpNormalizedLandmark landmarks[1] = {{.x = 0.1f, .y = 0.2f, .z = 0.3f}};
  MpNormalizedLandmarks landmarkList = {.landmarks = landmarks, .landmarks_count = 1};

  char categoryName[] = "brow";
  char displayName[] = "";
  MpCategory category = {
      .index = 0, .score = 0.9f, .category_name = categoryName, .display_name = displayName};
  MpCategories categories = {.categories = &category, .categories_count = 1};

  float matrixData[16] = {0};
  MpMatrix matrix = {.rows = 4, .cols = 4, .data = matrixData};

  MpFaceLandmarkerResult cResult = {
      .face_landmarks = &landmarkList,
      .face_landmarks_count = 1,
      .face_blendshapes = &categories,
      .face_blendshapes_count = 1,
      .facial_transformation_matrixes = &matrix,
      .facial_transformation_matrixes_count = 1,
  };

  MPPFaceLandmarkerResult *results =
      [MPPFaceLandmarkerResult faceLandmarkerResultWithCResult:cResult timestampInMilliseconds:42];

  XCTAssertEqual(results.faceLandmarks.count, 1);
  XCTAssertEqual(results.faceLandmarks[0].count, 1);
  XCTAssertEqualWithAccuracy(results.faceLandmarks[0][0].x, 0.1f, 1e-4f);
  XCTAssertEqualWithAccuracy(results.faceLandmarks[0][0].y, 0.2f, 1e-4f);
  XCTAssertEqualWithAccuracy(results.faceLandmarks[0][0].z, 0.3f, 1e-4f);

  XCTAssertEqual(results.faceBlendshapes.count, 1);
  XCTAssertEqual(results.faceBlendshapes[0].headIndex, 0);
  XCTAssertEqualObjects(results.faceBlendshapes[0].headName, @"");
  XCTAssertEqual(results.faceBlendshapes[0].categories.count, 1);
  XCTAssertEqual(results.faceBlendshapes[0].categories[0].index, 0);
  XCTAssertEqualWithAccuracy(results.faceBlendshapes[0].categories[0].score, 0.9f, 1e-4f);
  XCTAssertEqualObjects(results.faceBlendshapes[0].categories[0].categoryName, @"brow");
  XCTAssertEqualObjects(results.faceBlendshapes[0].categories[0].displayName, @"");

  XCTAssertEqual(results.facialTransformationMatrixes.count, 1);
  XCTAssertEqual(results.timestampInMilliseconds, 42);
}

- (void)testCreatesCopyOfFacialTransformationMatrix {
  float matrixData[16];
  for (size_t i = 0; i < 16; ++i) {
    matrixData[i] = 0.1f * i;
  }
  MpMatrix matrix = {.rows = 4, .cols = 4, .data = matrixData};

  MpFaceLandmarkerResult cResult = {
      .face_landmarks = nullptr,
      .face_landmarks_count = 0,
      .face_blendshapes = nullptr,
      .face_blendshapes_count = 0,
      .facial_transformation_matrixes = &matrix,
      .facial_transformation_matrixes_count = 1,
  };

  MPPFaceLandmarkerResult *results =
      [MPPFaceLandmarkerResult faceLandmarkerResultWithCResult:cResult timestampInMilliseconds:0];
  for (float &value : matrixData) {
    value = 0.0f;
  }

  XCTAssertEqual(results.facialTransformationMatrixes.count, 1);
  XCTAssertEqual(results.facialTransformationMatrixes[0].rows, 4);
  XCTAssertEqual(results.facialTransformationMatrixes[0].columns, 4);
  for (size_t column = 0; column < 4; ++column) {
    for (size_t row = 0; row < 4; ++row) {
      XCTAssertEqualWithAccuracy(
          [results.facialTransformationMatrixes[0] valueAtRow:row column:column],
          0.4f * row + 0.1f * column, /* accuracy= */ 0.0001f, @"at [%zu,%zu]", column, row);
    }
  }
}

- (void)testCreatesResultFromEmptyCResult {
  MpFaceLandmarkerResult emptyResult = {};
  MPPFaceLandmarkerResult *results =
      [MPPFaceLandmarkerResult faceLandmarkerResultWithCResult:emptyResult
                                       timestampInMilliseconds:0];

  NSArray *emptyArray = [NSArray array];
  XCTAssertEqualObjects(results.faceLandmarks, emptyArray);
  XCTAssertEqualObjects(results.faceBlendshapes, emptyArray);
  XCTAssertEqualObjects(results.facialTransformationMatrixes, emptyArray);
  XCTAssertEqual(results.timestampInMilliseconds, 0);
}

@end
