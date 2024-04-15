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

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <XCTest/XCTest.h>

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"
#import "mediapipe/tasks/ios/vision/face_stylizer/sources/MPPFaceStylizer.h"
#import "mediapipe/tasks/ios/vision/face_stylizer/sources/MPPFaceStylizerResult.h"

static MPPFileInfo *const kFaceStylizerBundleAssetFileInfo =
    [[MPPFileInfo alloc] initWithName:@"face_stylizer_color_ink" type:@"task"];

static MPPFileInfo *const kLargeFaceImageFileInfo = [[MPPFileInfo alloc] initWithName:@"portrait"
                                                                                 type:@"jpg"];
static MPPFileInfo *const kSmallFaceImageFileInfo =
    [[MPPFileInfo alloc] initWithName:@"portrait_small" type:@"jpg"];

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";
static const NSInteger kModelImageSize = 256;

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

#define AssertFaceStylizerResultProperties(result, kExpectedStylizedImageSize) \
  XCTAssertNotNil(result);                                                     \
  XCTAssertNotNil(result.stylizedImage);                                       \
  XCTAssertEqual(result.stylizedImage.width, kExpectedStylizedImageSize);      \
  XCTAssertEqual(result.stylizedImage.height, kExpectedStylizedImageSize);

@interface MPPFaceStylizerTests : XCTestCase
@end

@implementation MPPFaceStylizerTests

#pragma mark General Tests

- (void)testStylizeWithModelPathSucceeds {
  NSError *error = nil;

  MPPFaceStylizer *faceStylizer =
      [[MPPFaceStylizer alloc] initWithModelPath:kFaceStylizerBundleAssetFileInfo.path
                                           error:&error];
  XCTAssertNotNil(faceStylizer);
  XCTAssertNil(error);

  [self assertResultOfStylizeImageWithFileInfo:kLargeFaceImageFileInfo
                             usingFaceStylizer:faceStylizer
                                       hasSize:kModelImageSize];
}

- (void)testStylizeWithOptionsImageSucceeds {
  MPPFaceStylizer *faceStylizer =
      [self createFaceStylizerFromOptionsWithModelFileInfo:kFaceStylizerBundleAssetFileInfo];

  [self assertResultOfStylizeImageWithFileInfo:kLargeFaceImageFileInfo
                             usingFaceStylizer:faceStylizer
                                       hasSize:kModelImageSize];
}

- (void)testStylizeSmallImageSucceeds {
  MPPFaceStylizer *faceStylizer =
      [self createFaceStylizerFromOptionsWithModelFileInfo:kFaceStylizerBundleAssetFileInfo];

  [self assertResultOfStylizeImageWithFileInfo:kSmallFaceImageFileInfo
                             usingFaceStylizer:faceStylizer
                                       hasSize:kModelImageSize];
}

- (void)testStylizeWithNoFaceInImageSucceeds {
  MPPFaceStylizer *faceStylizer =
      [self createFaceStylizerFromOptionsWithModelFileInfo:kFaceStylizerBundleAssetFileInfo];
  MPPImage *image = [self createImageWithFileInfo:kLargeFaceImageFileInfo];

  CGRect rect = CGRectMake(0.1f, 0.1f, 0.1f, 0.1f);

  MPPFaceStylizerResult *result = [faceStylizer stylizeImage:image regionOfInterest:rect error:nil];
  XCTAssertNotNil(result);
  XCTAssertNil(result.stylizedImage);
}

- (void)testStylizeWithRegionOfInterestSucceeds {
  MPPFaceStylizer *faceStylizer =
      [self createFaceStylizerFromOptionsWithModelFileInfo:kFaceStylizerBundleAssetFileInfo];
  MPPImage *image = [self createImageWithFileInfo:kLargeFaceImageFileInfo];

  CGRect rect = CGRectMake(0.32f, 0.02f, 0.35f, 0.3f);

  MPPFaceStylizerResult *result = [faceStylizer stylizeImage:image regionOfInterest:rect error:nil];
  AssertFaceStylizerResultProperties(result, kModelImageSize);
}

- (void)testCreateImageClassifierWithMissingModelPathFails {
  MPPFileInfo *missingFileInfo = [[MPPFileInfo alloc] initWithName:@""
                                                                        type:@""];

  NSError *error = nil;
  MPPFaceStylizer *faceStylizer = [[MPPFaceStylizer alloc] initWithModelPath:missingFileInfo.path
                                                                                error:&error];
  XCTAssertNil(faceStylizer);

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

#pragma mark Face Stylizer Initializers

- (MPPFaceStylizer *)createFaceStylizerFromOptionsWithModelFileInfo:(MPPFileInfo *)modelFileInfo {
  MPPFaceStylizerOptions *faceStylizerOptions = [[MPPFaceStylizerOptions alloc] init];
  faceStylizerOptions.baseOptions.modelAssetPath = modelFileInfo.path;

  MPPFaceStylizer *faceStylizer = [[MPPFaceStylizer alloc] initWithOptions:faceStylizerOptions
                                                                     error:nil];
  XCTAssertNotNil(faceStylizer);

  return faceStylizer;
}

#pragma mark Face Stylizer Results

- (void)assertResultOfStylizeImageWithFileInfo:(MPPFileInfo *)fileInfo
                             usingFaceStylizer:(MPPFaceStylizer *)faceStylizer
                                       hasSize:(NSInteger)modelImageSize {
  MPPImage *image = [self createImageWithFileInfo:fileInfo];

  MPPFaceStylizerResult *result = [faceStylizer stylizeImage:image error:nil];
  AssertFaceStylizerResultProperties(result, kModelImageSize);
}

#pragma mark MPImage Initializers

- (MPPImage *)createImageWithFileInfo:(MPPFileInfo *)imageFileInfo {
  MPPImage *image = [MPPImage imageWithFileInfo:imageFileInfo];
  XCTAssertNotNil(image);

  return image;
}

@end
