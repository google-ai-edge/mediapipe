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

#import "mediapipe/tasks/ios/vision/image_classifier/sources/MPPImageClassifier.h"

#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionPacketCreator.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/image_classifier/utils/sources/MPPImageClassifierOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/image_classifier/utils/sources/MPPImageClassifierResult+Helpers.h"

#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"

namespace {
using ::mediapipe::NormalizedRect;
using ::mediapipe::Packet;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // namespace

static NSString *const kClassificationsStreamName = @"classifications_out";
static NSString *const kClassificationsTag = @"CLASSIFICATIONS";
static NSString *const kImageInStreamName = @"image_in";
static NSString *const kImageOutStreamName = @"image_out";
static NSString *const kImageTag = @"IMAGE";
static NSString *const kNormRectStreamName = @"norm_rect_in";
static NSString *const kNormRectTag = @"NORM_RECT";

static NSString *const kTaskGraphName =
    @"mediapipe.tasks.vision.image_classifier.ImageClassifierGraph";

#define InputPacketMap(imagePacket, normalizedRectPacket) \
  {                                                       \
    {kImageInStreamName.cppString, imagePacket}, {        \
      kNormRectStreamName.cppString, normalizedRectPacket \
    }                                                     \
  }

@interface MPPImageClassifier () {
  /** iOS Vision Task Runner */
  MPPVisionTaskRunner *_visionTaskRunner;
}
@end

@implementation MPPImageClassifier

- (instancetype)initWithOptions:(MPPImageClassifierOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
        initWithTaskGraphName:kTaskGraphName
                 inputStreams:@[
                   [NSString stringWithFormat:@"%@:%@", kImageTag, kImageInStreamName],
                   [NSString stringWithFormat:@"%@:%@", kNormRectTag, kNormRectStreamName]
                 ]
                outputStreams:@[
                  [NSString
                      stringWithFormat:@"%@:%@", kClassificationsTag, kClassificationsStreamName],
                  [NSString stringWithFormat:@"%@:%@", kImageTag, kImageOutStreamName]
                ]
                  taskOptions:options
           enableFlowLimiting:options.runningMode == MPPRunningModeLiveStream
                        error:error];

    if (!taskInfo) {
      return nil;
    }

    PacketsCallback packetsCallback = nullptr;

    if (options.completion) {
      packetsCallback = [=](absl::StatusOr<PacketMap> status_or_packets) {
        NSError *callbackError = nil;
        MPPImageClassifierResult *result;
        if ([MPPCommonUtils checkCppError:status_or_packets.status() toError:&callbackError]) {
          result = [MPPImageClassifierResult
              imageClassifierResultWithClassificationsPacket:
                  status_or_packets.value()[kClassificationsStreamName.cppString]];
        }
        options.completion(result, callbackError);
      };
    }

    _visionTaskRunner =
        [[MPPVisionTaskRunner alloc] initWithCalculatorGraphConfig:[taskInfo generateGraphConfig]
                                                       runningMode:options.runningMode
                                                   packetsCallback:std::move(packetsCallback)
                                                             error:error];

    if (!_visionTaskRunner) {
      return nil;
    }
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPImageClassifierOptions *options = [[MPPImageClassifierOptions alloc] init];

  options.baseOptions.modelAssetPath = modelPath;

  return [self initWithOptions:options error:error];
}

- (nullable MPPImageClassifierResult *)classifyImage:(MPPImage *)image
                                    regionOfInterest:(CGRect)roi
                                               error:(NSError **)error {
  std::optional<NormalizedRect> rect =
      [_visionTaskRunner normalizedRectFromRegionOfInterest:roi
                                                  imageSize:CGSizeMake(image.width, image.height)
                                           imageOrientation:image.orientation
                                                 ROIAllowed:YES
                                                      error:error];
  if (!rect.has_value()) {
    return nil;
  }

  Packet imagePacket = [MPPVisionPacketCreator createPacketWithMPPImage:image error:error];
  if (imagePacket.IsEmpty()) {
    return nil;
  }

  Packet normalizedRectPacket =
      [MPPVisionPacketCreator createPacketWithNormalizedRect:rect.value()];

  PacketMap inputPacketMap = InputPacketMap(imagePacket, normalizedRectPacket);

  std::optional<PacketMap> outputPacketMap = [_visionTaskRunner processImagePacketMap:inputPacketMap
                                                                                error:error];
  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return
      [MPPImageClassifierResult imageClassifierResultWithClassificationsPacket:
                                    outputPacketMap.value()[kClassificationsStreamName.cppString]];
}

- (std::optional<PacketMap>)inputPacketMapWithMPPImage:(MPPImage *)image
                               timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                      regionOfInterest:(CGRect)roi
                                                 error:(NSError **)error {
  std::optional<NormalizedRect> rect =
      [_visionTaskRunner normalizedRectFromRegionOfInterest:roi
                                                  imageSize:CGSizeMake(image.width, image.height)
                                           imageOrientation:image.orientation
                                                 ROIAllowed:YES
                                                      error:error];
  if (!rect.has_value()) {
    return std::nullopt;
  }

  Packet imagePacket = [MPPVisionPacketCreator createPacketWithMPPImage:image
                                                timestampInMilliseconds:timestampInMilliseconds
                                                                  error:error];
  if (imagePacket.IsEmpty()) {
    return std::nullopt;
  }

  Packet normalizedRectPacket =
      [MPPVisionPacketCreator createPacketWithNormalizedRect:rect.value()
                                     timestampInMilliseconds:timestampInMilliseconds];

  PacketMap inputPacketMap = InputPacketMap(imagePacket, normalizedRectPacket);
  return inputPacketMap;
}

- (nullable MPPImageClassifierResult *)classifyImage:(MPPImage *)image error:(NSError **)error {
  return [self classifyImage:image regionOfInterest:CGRectZero error:error];
}

- (nullable MPPImageClassifierResult *)classifyVideoFrame:(MPPImage *)image
                                  timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                         regionOfInterest:(CGRect)roi
                                                    error:(NSError **)error {
  std::optional<PacketMap> inputPacketMap = [self inputPacketMapWithMPPImage:image
                                                     timestampInMilliseconds:timestampInMilliseconds
                                                            regionOfInterest:roi
                                                                       error:error];
  if (!inputPacketMap.has_value()) {
    return nil;
  }

  std::optional<PacketMap> outputPacketMap =
      [_visionTaskRunner processVideoFramePacketMap:inputPacketMap.value() error:error];

  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return
      [MPPImageClassifierResult imageClassifierResultWithClassificationsPacket:
                                    outputPacketMap.value()[kClassificationsStreamName.cppString]];
}

- (nullable MPPImageClassifierResult *)classifyVideoFrame:(MPPImage *)image
                                  timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                    error:(NSError **)error {
  return [self classifyVideoFrame:image
          timestampInMilliseconds:timestampInMilliseconds
                 regionOfInterest:CGRectZero
                            error:error];
}

- (BOOL)classifyAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
           regionOfInterest:(CGRect)roi
                      error:(NSError **)error {
  std::optional<PacketMap> inputPacketMap = [self inputPacketMapWithMPPImage:image
                                                     timestampInMilliseconds:timestampInMilliseconds
                                                            regionOfInterest:roi
                                                                       error:error];
  if (!inputPacketMap.has_value()) {
    return NO;
  }

  return [_visionTaskRunner processLiveStreamPacketMap:inputPacketMap.value() error:error];
}

- (BOOL)classifyAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  return [self classifyAsyncImage:image
          timestampInMilliseconds:timestampInMilliseconds
                 regionOfInterest:CGRectZero
                            error:error];
}

@end
