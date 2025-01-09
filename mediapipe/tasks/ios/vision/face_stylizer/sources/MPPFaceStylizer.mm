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

#import "mediapipe/tasks/ios/vision/face_stylizer/sources/MPPFaceStylizer.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/face_stylizer/utils/sources/MPPFaceStylizerOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/face_stylizer/utils/sources/MPPFaceStylizerResult+Helpers.h"

namespace {
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // namespace

static NSString *const kImageTag = @"IMAGE";
static NSString *const kImageInStreamName = @"image_in";
static NSString *const kNormRectTag = @"NORM_RECT";
static NSString *const kNormRectInStreamName = @"norm_rect_in";
static NSString *const kStylizedImageTag = @"STYLIZED_IMAGE";
static NSString *const kStylizedImageOutStreamName = @"stylized_image";
static NSString *const kTaskGraphName = @"mediapipe.tasks.vision.face_stylizer.FaceStylizerGraph";
static NSString *const kTaskName = @"faceStylizer";

@interface MPPFaceStylizer () {
  /** iOS Vision Task Runner */
  MPPVisionTaskRunner *_visionTaskRunner;
}

@end

@implementation MPPFaceStylizer

#pragma mark - Public

- (instancetype)initWithOptions:(MPPFaceStylizerOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
        initWithTaskGraphName:kTaskGraphName
                 inputStreams:@[
                   [NSString stringWithFormat:@"%@:%@", kImageTag, kImageInStreamName],
                   [NSString stringWithFormat:@"%@:%@", kNormRectTag, kNormRectInStreamName]
                 ]
                outputStreams:@[
                  [NSString
                      stringWithFormat:@"%@:%@", kStylizedImageTag, kStylizedImageOutStreamName],
                ]
                  taskOptions:options
           enableFlowLimiting:NO
                        error:error];

    if (!taskInfo) {
      return nil;
    }

    PacketsCallback packetsCallback = nullptr;

    _visionTaskRunner = [[MPPVisionTaskRunner alloc] initWithTaskInfo:taskInfo
                                                          runningMode:MPPRunningModeImage
                                                           roiAllowed:YES
                                                      packetsCallback:nullptr
                                                 imageInputStreamName:kImageInStreamName
                                              normRectInputStreamName:kNormRectInStreamName
                                                                error:error];
    if (!_visionTaskRunner) {
      return nil;
    }
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPFaceStylizerOptions *options = [[MPPFaceStylizerOptions alloc] init];

  options.baseOptions.modelAssetPath = modelPath;

  return [self initWithOptions:options error:error];
}

- (nullable MPPFaceStylizerResult *)stylizeImage:(MPPImage *)image error:(NSError **)error {
  return [self stylizeImage:image
               regionOfInterest:CGRectZero
      shouldCopyOutputPixelData:YES
                          error:error];
}

- (nullable MPPFaceStylizerResult *)stylizeImage:(MPPImage *)image
                                regionOfInterest:(CGRect)regionOfInterest
                                           error:(NSError **)error {
  return [self stylizeImage:image
               regionOfInterest:regionOfInterest
      shouldCopyOutputPixelData:YES
                          error:error];
}

#pragma mark - Private

- (MPPFaceStylizerResult *)stylizeImage:(MPPImage *)image
                       regionOfInterest:(CGRect)regionOfInterest
              shouldCopyOutputPixelData:(BOOL)shouldCopyOutputPixelData
                                  error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap = [_visionTaskRunner processImage:image
                                                            regionOfInterest:regionOfInterest
                                                                       error:error];

  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return [MPPFaceStylizerResult
      faceStylizerResultWithStylizedImagePacket:outputPacketMap
                                                    .value()[kStylizedImageOutStreamName.cppString]
                                    sourceImage:image
                            shouldCopyPixelData:shouldCopyOutputPixelData
                                          error:error];
}

@end
