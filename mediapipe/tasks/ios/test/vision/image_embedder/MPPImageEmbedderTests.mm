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
#import <XCTest/XCTest.h>

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/test/vision/utils/sources/MPPImage+TestUtils.h"
#import "mediapipe/tasks/ios/vision/image_embedder/sources/MPPImageEmbedder.h"
#import "mediapipe/tasks/ios/vision/image_embedder/sources/MPPImageEmbedderResult.h"

#include <iostream>
#include <vector>

static MPPFileInfo *const kBurgerImageFileInfo = [[MPPFileInfo alloc] initWithName:@"burger"
                                                                              type:@"jpg"];
static MPPFileInfo *const kBurgerCroppedImageFileInfo =
    [[MPPFileInfo alloc] initWithName:@"burger_crop" type:@"jpg"];
static MPPFileInfo *const kBurgerRotatedImageFileInfo =
    [[MPPFileInfo alloc] initWithName:@"burger_rotated" type:@"jpg"];

static MPPFileInfo *const kMobileNetEmbedderModelFileInfo =
    [[MPPFileInfo alloc] initWithName:@"mobilenet_v3_small_100_224_embedder" type:@"tflite"];

static NSString *const kExpectedErrorDomain = @"com.google.mediapipe.tasks";

constexpr double kDoubleDifferenceTolerance = 1e-4;
constexpr NSInteger kExpectedEmbeddingLength = 1024;

#define AssertEqualErrors(error, expectedError)              \
  XCTAssertNotNil(error);                                    \
  XCTAssertEqualObjects(error.domain, expectedError.domain); \
  XCTAssertEqual(error.code, expectedError.code);            \
  XCTAssertEqualObjects(error.localizedDescription, expectedError.localizedDescription)

#define AssertImageEmbedderResultHasOneEmbedding(imageEmbedderResult) \
  XCTAssertNotNil(imageEmbedderResult);                               \
  XCTAssertNotNil(imageEmbedderResult.embeddingResult);               \
  \ 
  XCTAssertEqual(imageEmbedderResult.embeddingResult.embeddings.count, 1);

#define AssertEmbeddingHasCorrectTypeAndDimension(embedding, quantized, expectedLength) \
  if (quantized) {                                                                      \
    XCTAssertNil(embedding.floatEmbedding);                                             \
    XCTAssertNotNil(embedding.quantizedEmbedding);                                      \
    XCTAssertEqual(embedding.quantizedEmbedding.count, expectedLength);                 \
  } else {                                                                              \
    XCTAssertNotNil(embedding.floatEmbedding);                                          \
    XCTAssertNil(embedding.quantizedEmbedding);                                         \
    XCTAssertEqual(embedding.floatEmbedding.count, expectedLength);                     \
  }

@interface MPPImageEmbedderTests : XCTestCase
@end

@implementation MPPImageEmbedderTests

#pragma mark General Tests

- (void)testCreateImageEmbedderFailsWithMissingModelPath {
  MPPFileInfo *fileInfo = [[MPPFileInfo alloc] initWithName:@"" type:@""];

  NSError *error = nil;
  MPPImageEmbedder *imageEmbedder = [[MPPImageEmbedder alloc] initWithModelPath:fileInfo.path
                                                                          error:&error];
  XCTAssertNil(imageEmbedder);

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

- (void)testEmbedWithNoOptionsSucceeds {
  MPPImageEmbedder *imageEmbedder =
      [[MPPImageEmbedder alloc] initWithModelPath:kMobileNetEmbedderModelFileInfo.path error:nil];

  [self assertResultsOfEmbedImageWithFileInfo:kBurgerImageFileInfo
      isApproximatelyEqualToEmbedImageWithFileInfo:kBurgerCroppedImageFileInfo
                                usingImageEmbedder:imageEmbedder
                                       isQuantized:NO
                                withExpectedLength:kExpectedEmbeddingLength
                               andCosineSimilarity:0.928711f];
}

- (void)testEmbedWithQuantizationSucceeds {
  MPPImageEmbedderOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kMobileNetEmbedderModelFileInfo];
  options.quantize = YES;

  MPPImageEmbedder *imageEmbedder = [self createImageEmbedderWithOptionsSucceeds:options];

  [self assertResultsOfEmbedImageWithFileInfo:kBurgerImageFileInfo
      isApproximatelyEqualToEmbedImageWithFileInfo:kBurgerCroppedImageFileInfo
                                usingImageEmbedder:imageEmbedder
                                       isQuantized:options.quantize
                                withExpectedLength:kExpectedEmbeddingLength
                               andCosineSimilarity:0.92883f];
}

- (void)testEmbedWithL2NormalizationSucceeds {
  MPPImageEmbedderOptions *options =
      [self imageSegmenterOptionsWithModelFileInfo:kMobileNetEmbedderModelFileInfo];
  options.l2Normalize = YES;

  MPPImageEmbedder *imageEmbedder = [self createImageEmbedderWithOptionsSucceeds:options];

  [self assertResultsOfEmbedImageWithFileInfo:kBurgerImageFileInfo
      isApproximatelyEqualToEmbedImageWithFileInfo:kBurgerCroppedImageFileInfo
                                usingImageEmbedder:imageEmbedder
                                       isQuantized:options.quantize
                                withExpectedLength:kExpectedEmbeddingLength
                               andCosineSimilarity:0.928711f];
}

- (void)testEmbedWithRegionOfInterestSucceeds {
  MPPImageEmbedder *imageEmbedder =
      [[MPPImageEmbedder alloc] initWithModelPath:kMobileNetEmbedderModelFileInfo.path error:nil];

  MPPImage *burgerImage = [self assertCreateImageWithFileInfo:kBurgerImageFileInfo];
  MPPImageEmbedderResult *burgerImageResult =
      [imageEmbedder embedImage:burgerImage
               regionOfInterest:CGRectMake(0.0f, 0.0f, 0.833333f, 1.0f)
                          error:nil];

  MPPImage *burgerCroppedImage = [self assertCreateImageWithFileInfo:kBurgerCroppedImageFileInfo];
  MPPImageEmbedderResult *burgerCroppedImageResult = [imageEmbedder embedImage:burgerCroppedImage
                                                                         error:nil];

  [self assertImageEmbedderResult:burgerImageResult
      isApproximatelyEqualToImageEmbedderResult:burgerCroppedImageResult
                                    isQuantized:NO
                             withExpectedLength:kExpectedEmbeddingLength
                            andCosineSimilarity:0.99992f];
}

- (void)testEmbedWithOrientationSucceeds {
  MPPImageEmbedder *imageEmbedder =
      [[MPPImageEmbedder alloc] initWithModelPath:kMobileNetEmbedderModelFileInfo.path error:nil];

  MPPImage *burgerRotatedImage = [self assertCreateImageWithFileInfo:kBurgerRotatedImageFileInfo
                                                         orientation:UIImageOrientationLeft];
  MPPImageEmbedderResult *burgerRotatedImageResult = [imageEmbedder embedImage:burgerRotatedImage
                                                                         error:nil];

  MPPImage *burgerImage = [self assertCreateImageWithFileInfo:kBurgerImageFileInfo];
  MPPImageEmbedderResult *burgerImageResult = [imageEmbedder embedImage:burgerImage error:nil];

  [self assertImageEmbedderResult:burgerRotatedImageResult
      isApproximatelyEqualToImageEmbedderResult:burgerImageResult
                                    isQuantized:NO
                             withExpectedLength:kExpectedEmbeddingLength
                            andCosineSimilarity:0.98086f];
}

#pragma mark - Image Embedder Initializers

- (MPPImageEmbedderOptions *)imageSegmenterOptionsWithModelFileInfo:(MPPFileInfo *)fileInfo {
  MPPImageEmbedderOptions *options = [[MPPImageEmbedderOptions alloc] init];
  options.baseOptions.modelAssetPath = fileInfo.path;
  return options;
}

- (MPPImageEmbedder *)createImageEmbedderWithOptionsSucceeds:(MPPImageEmbedderOptions *)options {
  NSError *error;
  MPPImageEmbedder *imageEmbedder = [[MPPImageEmbedder alloc] initWithOptions:options error:&error];
  XCTAssertNotNil(imageEmbedder);
  XCTAssertNil(error);

  return imageEmbedder;
}

- (void)assertCreateImageEmbedderWithOptions:(MPPImageEmbedderOptions *)options
                      failsWithExpectedError:(NSError *)expectedError {
  NSError *error = nil;
  MPPImageEmbedder *imageSegmenter = [[MPPImageEmbedder alloc] initWithOptions:options
                                                                         error:&error];

  XCTAssertNil(imageSegmenter);
  AssertEqualErrors(error, expectedError);
}

#pragma mark MPPImage Helpers

- (MPPImage *)assertCreateImageWithFileInfo:(MPPFileInfo *)imageFileInfo {
  MPPImage *image = [MPPImage imageWithFileInfo:imageFileInfo];
  XCTAssertNotNil(image);

  return image;
}

- (MPPImage *)assertCreateImageWithFileInfo:(MPPFileInfo *)imageFileInfo
                                orientation:(UIImageOrientation)orientation {
  MPPImage *image = [MPPImage imageWithFileInfo:imageFileInfo orientation:orientation];
  XCTAssertNotNil(image);

  return image;
}

#pragma mark Assert Embedder Results
- (void)assertResultsOfEmbedImageWithFileInfo:(MPPFileInfo *)firstImageFileInfo
    isApproximatelyEqualToEmbedImageWithFileInfo:(MPPFileInfo *)secondImageFileInfo
                              usingImageEmbedder:(MPPImageEmbedder *)imageEmbedder
                                     isQuantized:(BOOL)isQuantized
                              withExpectedLength:(NSInteger)expectedLength
                             andCosineSimilarity:(double)expectedCosineSimilarity {
  MPPImage *firstImage = [self assertCreateImageWithFileInfo:firstImageFileInfo];
  MPPImageEmbedderResult *firstEmbedderResult = [imageEmbedder embedImage:firstImage error:nil];

  MPPImage *secondImage = [self assertCreateImageWithFileInfo:secondImageFileInfo];
  MPPImageEmbedderResult *secondEmbedderResult = [imageEmbedder embedImage:secondImage error:nil];

  [self assertImageEmbedderResult:firstEmbedderResult
      isApproximatelyEqualToImageEmbedderResult:secondEmbedderResult
                                    isQuantized:isQuantized
                             withExpectedLength:expectedLength
                            andCosineSimilarity:expectedCosineSimilarity];
}

- (void)assertImageEmbedderResult:(MPPImageEmbedderResult *)firstImageEmbedderResult
    isApproximatelyEqualToImageEmbedderResult:(MPPImageEmbedderResult *)secondImageEmbedderResult
                                  isQuantized:(BOOL)isQuantized
                           withExpectedLength:(NSInteger)expectedLength
                          andCosineSimilarity:(double)expectedCosineSimilarity {
  AssertImageEmbedderResultHasOneEmbedding(firstImageEmbedderResult);
  AssertEmbeddingHasCorrectTypeAndDimension(firstImageEmbedderResult.embeddingResult.embeddings[0],
                                            isQuantized, expectedLength);

  AssertImageEmbedderResultHasOneEmbedding(secondImageEmbedderResult);
  AssertEmbeddingHasCorrectTypeAndDimension(secondImageEmbedderResult.embeddingResult.embeddings[0],
                                            isQuantized, expectedLength);

  NSNumber *cosineSimilarity = [MPPImageEmbedder
      cosineSimilarityBetweenEmbedding1:firstImageEmbedderResult.embeddingResult.embeddings[0]
                          andEmbedding2:secondImageEmbedderResult.embeddingResult.embeddings[0]
                                  error:nil];

  XCTAssertNotNil(cosineSimilarity);

  XCTAssertEqualWithAccuracy(cosineSimilarity.doubleValue, expectedCosineSimilarity,
                             kDoubleDifferenceTolerance);
}

@end
