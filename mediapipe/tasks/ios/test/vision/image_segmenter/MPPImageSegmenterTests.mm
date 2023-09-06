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
#import <XCTest/XCTest.h>

#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPMask+TestUtils.h"
#import "mediapipe/tasks/ios/vision/image_segmenter/sources/MPPImageSegmenter.h"
#import "mediapipe/tasks/ios/vision/image_segmenter/sources/MPPImageSegmenterResult.h"

static MPPFileInfo *const kCatImageFileInfo = [[MPPFileInfo alloc] initWithName:@"cat"
                                                                      type:@"jpg"];
static MPPFileInfo *const kCatGoldenImageFileInfo = [[MPPFileInfo alloc] initWithName:@"cat_mask"
                                                                            type:@"jpg"];
static MPPFileInfo *const kSegmentationImageFileInfo =
    [[MPPFileInfo alloc] initWithName:@"segmentation_input_rotation0" type:@"jpg"];
static MPPFileInfo *const kSegmentationGoldenImageFileInfo =
    [[MPPFileInfo alloc] initWithName:@"segmentation_golden_rotation0" type:@"png"];
static MPPFileInfo *const kImageSegmenterModel = [[MPPFileInfo alloc] initWithName:@"deeplabv3"
                                                                         type:@"tflite"];
static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";
constexpr float kSimilarityThreshold = 0.96f;
constexpr NSInteger kMagnificationFactor = 10;

double sum(const float *mask, size_t size) {
  double sum = 0.0;
  for (int i = 0; i < size; i++) {
    sum += mask[i];
  }
  return sum;
}

float *multiply(const float *mask1, const float *mask2, size_t size) {
  double sum = 0.0;
  float *multipliedMask = (float *)malloc(size * sizeof(float));
  if (!multipliedMask) {
    exit(-1);
  }
  for (int i = 0; i < size; i++) {
    multipliedMask[i] = mask1[i] * mask2[i];
  }

  return multipliedMask;
}

double softIOU(const float *mask1, const float *mask2, size_t size) {
  float *interSectionVector = multiply(mask1, mask2, size);
  double interSectionSum = sum(interSectionVector, size);
  free(interSectionVector);

  float *m1m1Vector = multiply(mask1, mask1, size);
  double m1m1 = sum(m1m1Vector, size);
  free(m1m1Vector);

  float *m2m2Vector = multiply(mask2, mask2, size);
  double m2m2 = sum(m2m2Vector, size);
  free(m2m2Vector);

  double unionSum = m1m1 + m2m2 - interSectionSum;

  return unionSum > 0.0 ? interSectionSum / unionSum : 0.0;
}

@interface MPPImageSegmenterTests : XCTestCase <MPPImageSegmenterLiveStreamDelegate> 

@end

@implementation MPPImageSegmenterTests

#pragma mark General Tests

- (void)setUp {
  // When expected and actual mask sizes are not equal, iterating through mask data results in a
  // segmentation fault. Setting this property to `NO`, prevents each test case from executing the
  // remaining flow after a failure. Since expected and actual mask sizes are compared before
  // iterating through them, this prevents any illegal memory access.
  self.continueAfterFailure = NO;
}

+ (NSString *)filePathWithName : (NSString *)fileName extension : (NSString *)extension {
  NSString *filePath =
      [[NSBundle bundleForClass:[MPPImageSegmenterTests class]] pathForResource:fileName
                                                                         ofType:extension];
  return filePath;
}

#pragma mark Image Mode Tests

- (void)testSegmentWithCategoryMaskSucceeds {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModel];
  options.shouldOutputConfidenceMasks = NO;
  options.shouldOutputCategoryMask = YES;

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  [self assertResultsOfSegmentImageWithFileInfo:kSegmentationImageFileInfo
                                           usingImageSegmenter:imageSegmenter
      approximatelyEqualsExpectedCategoryMaskImageWithFileInfo:kSegmentationGoldenImageFileInfo
                                     shouldHaveConfidenceMasks:NO];
}

- (void)testSegmentWithConfidenceMaskSucceeds {
  MPPImageSegmenterOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kImageSegmenterModel];

  MPPImageSegmenter *imageSegmenter = [self createImageSegmenterWithOptionsSucceeds:options];

  [self assertResultsOfSegmentImageWithFileInfo:kCatImageFileInfo
                                             usingImageSegmenter:imageSegmenter
      approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:kCatGoldenImageFileInfo
                                                         atIndex:8
                                          shouldHaveCategoryMask:NO];
}

#pragma mark - Image Segmenter Initializers

- (MPPImageSegmenterOptions *)imageSegmenterOptionsWithModelFileInfo:(MPPFileInfo *)fileInfo {
  MPPImageSegmenterOptions *options = [[MPPImageSegmenterOptions alloc] init];
  options.baseOptions.modelAssetPath = fileInfo.path;
  return options;
}

- (MPPImageSegmenter *)createImageSegmenterWithOptionsSucceeds:(MPPImageSegmenterOptions *)options {
  NSError *error;
  MPPImageSegmenter *imageSegmenter = [[MPPImageSegmenter alloc] initWithOptions:options
                                                                           error:&error];
  XCTAssertNotNil(imageSegmenter);
  XCTAssertNil(error);

  return imageSegmenter;
}

#pragma mark Assert Segmenter Results
- (void)assertResultsOfSegmentImageWithFileInfo:(MPPFileInfo *)imageFileInfo
                                         usingImageSegmenter:(MPPImageSegmenter *)imageSegmenter
    approximatelyEqualsExpectedCategoryMaskImageWithFileInfo:
        (MPPFileInfo *)expectedCategoryMaskFileInfo
                                   shouldHaveConfidenceMasks:(BOOL)shouldHaveConfidenceMasks {
  MPPImageSegmenterResult *result = [self segmentImageWithFileInfo:imageFileInfo
                                               usingImageSegmenter:imageSegmenter];

  XCTAssertNotNil(result.categoryMask);
  shouldHaveConfidenceMasks ? XCTAssertNotNil(result.confidenceMasks)
                            : XCTAssertNil(result.confidenceMasks);

  [self assertCategoryMask:result.categoryMask
      approximatelyEqualsExpectedCategoryMaskImageWithFileInfo:expectedCategoryMaskFileInfo];
}

- (void)assertResultsOfSegmentImageWithFileInfo:(MPPFileInfo *)imageFileInfo
                                           usingImageSegmenter:(MPPImageSegmenter *)imageSegmenter
    approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:
        (MPPFileInfo *)expectedConfidenceMaskFileInfo
                                                       atIndex:(NSInteger)index
                                        shouldHaveCategoryMask:(BOOL)shouldHaveCategoryMask {
  MPPImageSegmenterResult *result = [self segmentImageWithFileInfo:imageFileInfo
                                               usingImageSegmenter:imageSegmenter];

  XCTAssertNotNil(result.confidenceMasks);
  shouldHaveCategoryMask ? XCTAssertNotNil(result.categoryMask) : XCTAssertNil(result.categoryMask);

  XCTAssertLessThan(index, result.confidenceMasks.count);

  [self assertConfidenceMask:result.confidenceMasks[index]
      approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:expectedConfidenceMaskFileInfo];
}

- (MPPImageSegmenterResult *)segmentImageWithFileInfo:(MPPFileInfo *)fileInfo
                                  usingImageSegmenter:(MPPImageSegmenter *)imageSegmenter {
  MPPImage *image = [MPPImage imageWithFileInfo:fileInfo];
  XCTAssertNotNil(image);

  NSError *error;
  MPPImageSegmenterResult *result = [imageSegmenter segmentImage:image error:&error];
  XCTAssertNil(error);
  XCTAssertNotNil(result);

  return result;
}

- (void)assertCategoryMask:(MPPMask *)categoryMask
    approximatelyEqualsExpectedCategoryMaskImageWithFileInfo:
        (MPPFileInfo *)expectedCategoryMaskImageFileInfo {
  MPPMask *expectedCategoryMask =
      [[MPPMask alloc] initWithImageFileInfo:expectedCategoryMaskImageFileInfo];

  XCTAssertEqual(categoryMask.width, expectedCategoryMask.width);
  XCTAssertEqual(categoryMask.height, expectedCategoryMask.height);

  size_t maskSize = categoryMask.width * categoryMask.height;

  const UInt8 *categoryMaskPixelData = categoryMask.uint8Data;
  const UInt8 *expectedCategoryMaskPixelData = expectedCategoryMask.uint8Data;

  NSInteger consistentPixels = 0;

  for (int i = 0; i < maskSize; i++) {
    consistentPixels +=
        categoryMaskPixelData[i] * kMagnificationFactor == expectedCategoryMaskPixelData[i] ? 1 : 0;
  }

  XCTAssertGreaterThan((float)consistentPixels / (float)maskSize, kSimilarityThreshold);
}

- (void)assertConfidenceMask:(MPPMask *)confidenceMask
    approximatelyEqualsExpectedConfidenceMaskImageWithFileInfo:
        (MPPFileInfo *)expectedConfidenceMaskImageFileInfo {
  MPPMask *expectedConfidenceMask =
      [[MPPMask alloc] initWithImageFileInfo:expectedConfidenceMaskImageFileInfo];

  XCTAssertEqual(confidenceMask.width, expectedConfidenceMask.width);
  XCTAssertEqual(confidenceMask.height, expectedConfidenceMask.height);

  size_t maskSize = confidenceMask.width * confidenceMask.height;

  XCTAssertGreaterThan(
      softIOU(confidenceMask.float32Data, expectedConfidenceMask.float32Data, maskSize),
      kSimilarityThreshold);
}

@end
