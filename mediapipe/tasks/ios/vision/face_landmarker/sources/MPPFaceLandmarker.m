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

#import "mediapipe/tasks/ios/vision/face_landmarker/sources/MPPFaceLandmarker.h"
#import <Foundation/Foundation.h>

#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionPacketCreator.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/utils/sources/MPPFaceLandmarkerOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/utils/sources/MPPFaceLandmarkerResult+Helpers.h"

using ::mediapipe::NormalizedRect;
using ::mediapipe::Packet;
using ::mediapipe::Timestamp;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;

static constexpr int kMicrosecondsPerMillisecond = 1000;

// Constants for the underlying MP Tasks Graph. See
// https://github.com/google/mediapipe/tree/master/mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_graph.cc
static NSString *const kLandmarksOutStreamName = @"landmarks_out";
static NSString *const kLandmarksOutTag = @"NORM_LANDMARKS";
static NSString *const kBlendshapesOutStreamName = @"blendshapes_out";
static NSString *const kBlendshapesOutTag = @"BLENDSHAPES";
static NSString *const kFaceGeometryOutStreamName = @"face_geometry_out";
static NSString *const kFaceGeometryOutTag = @"FACE_GEOMETRY";
static NSString *const kNormRectStreamName = @"norm_rect_in";
static NSString *const kNormRectTag = @"NORM_RECT";
static NSString *const kImageInStreamName = @"image_in";
static NSString *const kImageOutStreamName = @"image_out";
static NSString *const kImageTag = @"IMAGE";
static NSString *const kTaskGraphName =
    @"mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph";
static NSString *const kTaskName = @"faceLandmarker";

#define InputPacketMap(imagePacket, normalizedRectPacket) \
  {                                                       \
    {kImageInStreamName.cppString, imagePacket}, {        \
      kNormRectStreamName.cppString, normalizedRectPacket \
    }                                                     \
  }

@interface MPPFaceLandmarker () {
  /** iOS Vision Task Runner */
  MPPVisionTaskRunner *_visionTaskRunner;
  /**
   * The callback queue for the live stream delegate. This is only set if the user provides a live
   * stream delegate.
   */
  dispatch_queue_t _callbackQueue;
  /** The user-provided live stream delegate if set. */
  __weak id<MPPFaceLandmarkerLiveStreamDelegate> _faceLandmarkerLiveStreamDelegate;
}
@end

@implementation MPPFaceLandmarker

- (instancetype)initWithOptions:(MPPFaceLandmarkerOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    NSArray<NSString *> *inputStreams = @[
      [NSString stringWithFormat:@"%@:%@", kImageTag, kImageInStreamName],
      [NSString stringWithFormat:@"%@:%@", kNormRectTag, kNormRectStreamName]
    ];

    NSMutableArray<NSString *> *outputStreams = [NSMutableArray
        arrayWithObjects:[NSString
                             stringWithFormat:@"%@:%@", kLandmarksOutTag, kLandmarksOutStreamName],
                         [NSString stringWithFormat:@"%@:%@", kImageTag, kImageOutStreamName], nil];
    if (options.outputFaceBlendshapes) {
      [outputStreams addObject:[NSString stringWithFormat:@"%@:%@", kBlendshapesOutTag,
                                                          kBlendshapesOutStreamName]];
    }
    if (options.outputFacialTransformationMatrixes) {
      [outputStreams addObject:[NSString stringWithFormat:@"%@:%@", kFaceGeometryOutTag,
                                                          kFaceGeometryOutStreamName]];
    }

    MPPTaskInfo *taskInfo =
        [[MPPTaskInfo alloc] initWithTaskGraphName:kTaskGraphName
                                      inputStreams:inputStreams
                                     outputStreams:outputStreams
                                       taskOptions:options
                                enableFlowLimiting:options.runningMode == MPPRunningModeLiveStream
                                             error:error];

    if (!taskInfo) {
      return nil;
    }

    PacketsCallback packetsCallback = nullptr;

    if (options.faceLandmarkerLiveStreamDelegate) {
      _faceLandmarkerLiveStreamDelegate = options.faceLandmarkerLiveStreamDelegate;

      // Create a private serial dispatch queue in which the delegate method will be called
      // asynchronously. This is to ensure that if the client performs a long running operation in
      // the delegate method, the queue on which the C++ callbacks is invoked is not blocked and is
      // freed up to continue with its operations.
      _callbackQueue = dispatch_queue_create(
          [MPPVisionTaskRunner uniqueDispatchQueueNameWithSuffix:kTaskName], NULL);

      // Capturing `self` as weak in order to avoid `self` being kept in memory
      // and cause a retain cycle, after self is set to `nil`.
      MPPFaceLandmarker *__weak weakSelf = self;
      packetsCallback = [weakSelf](absl::StatusOr<PacketMap> liveStreamResult) {
        [weakSelf processLiveStreamResult:liveStreamResult];
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
  MPPFaceLandmarkerOptions *options = [[MPPFaceLandmarkerOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (std::optional<PacketMap>)inputPacketMapWithMPPImage:(MPPImage *)image
                               timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                 error:(NSError **)error {
  std::optional<NormalizedRect> rect =
      [_visionTaskRunner normalizedRectWithImageOrientation:image.orientation
                                                  imageSize:CGSizeMake(image.width, image.height)
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
      [MPPVisionPacketCreator createPacketWithNormalizedRect:*rect
                                     timestampInMilliseconds:timestampInMilliseconds];

  PacketMap inputPacketMap = InputPacketMap(imagePacket, normalizedRectPacket);
  return inputPacketMap;
}

- (nullable MPPFaceLandmarkerResult *)detectInImage:(MPPImage *)image error:(NSError **)error {
  std::optional<NormalizedRect> rect =
      [_visionTaskRunner normalizedRectWithImageOrientation:image.orientation
                                                  imageSize:CGSizeMake(image.width, image.height)
                                                      error:error];
  if (!rect.has_value()) {
    return nil;
  }

  Packet imagePacket = [MPPVisionPacketCreator createPacketWithMPPImage:image error:error];
  if (imagePacket.IsEmpty()) {
    return nil;
  }

  Packet normalizedRectPacket = [MPPVisionPacketCreator createPacketWithNormalizedRect:*rect];

  PacketMap inputPacketMap = InputPacketMap(imagePacket, normalizedRectPacket);

  std::optional<PacketMap> outputPacketMap = [_visionTaskRunner processImagePacketMap:inputPacketMap
                                                                                error:error];
  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return [MPPFaceLandmarkerResult
      faceLandmarkerResultWithLandmarksPacket:outputPacketMap
                                                  .value()[kLandmarksOutStreamName.cppString]
                            blendshapesPacket:outputPacketMap
                                                  .value()[kBlendshapesOutStreamName.cppString]
                 transformationMatrixesPacket:outputPacketMap
                                                  .value()[kFaceGeometryOutStreamName.cppString]];
}

- (nullable MPPFaceLandmarkerResult *)detectInVideoFrame:(MPPImage *)image
                                 timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                   error:(NSError **)error {
  std::optional<PacketMap> inputPacketMap = [self inputPacketMapWithMPPImage:image
                                                     timestampInMilliseconds:timestampInMilliseconds
                                                                       error:error];
  if (!inputPacketMap.has_value()) {
    return nil;
  }

  std::optional<PacketMap> outputPacketMap =
      [_visionTaskRunner processVideoFramePacketMap:*inputPacketMap error:error];
  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return [MPPFaceLandmarkerResult
      faceLandmarkerResultWithLandmarksPacket:outputPacketMap
                                                  .value()[kLandmarksOutStreamName.cppString]
                            blendshapesPacket:outputPacketMap
                                                  .value()[kBlendshapesOutStreamName.cppString]
                 transformationMatrixesPacket:outputPacketMap
                                                  .value()[kFaceGeometryOutStreamName.cppString]];
}

- (BOOL)detectAsyncInImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  std::optional<PacketMap> inputPacketMap = [self inputPacketMapWithMPPImage:image
                                                     timestampInMilliseconds:timestampInMilliseconds
                                                                       error:error];
  if (!inputPacketMap.has_value()) {
    return NO;
  }

  return [_visionTaskRunner processLiveStreamPacketMap:*inputPacketMap error:error];
}

- (void)processLiveStreamResult:(absl::StatusOr<PacketMap>)liveStreamResult {
  NSError *callbackError;
  if (![MPPCommonUtils checkCppError:liveStreamResult.status() toError:&callbackError]) {
    dispatch_async(_callbackQueue, ^{
      [_faceLandmarkerLiveStreamDelegate faceLandmarker:self
                           didFinishDetectionWithResult:nil
                                timestampInMilliseconds:Timestamp::Unset().Value()
                                                  error:callbackError];
    });
    return;
  }

  PacketMap &outputPacketMap = *liveStreamResult;
  if (outputPacketMap[kImageOutStreamName.cppString].IsEmpty()) {
    // The graph did not return a result. We therefore do not raise the user callback. This mirrors
    // returning `nil` in the other methods and is acceptable for the live stream delegate since
    // it is expected that we drop frames and don't return results for every input.
    return;
  }

  MPPFaceLandmarkerResult *result = [MPPFaceLandmarkerResult
      faceLandmarkerResultWithLandmarksPacket:outputPacketMap[kLandmarksOutStreamName.cppString]
                            blendshapesPacket:outputPacketMap[kBlendshapesOutStreamName.cppString]
                 transformationMatrixesPacket:outputPacketMap[kFaceGeometryOutStreamName
                                                                  .cppString]];

  NSInteger timeStampInMilliseconds =
      outputPacketMap[kImageOutStreamName.cppString].Timestamp().Value() /
      kMicrosecondsPerMillisecond;
  dispatch_async(_callbackQueue, ^{
    [_faceLandmarkerLiveStreamDelegate faceLandmarker:self
                         didFinishDetectionWithResult:result
                              timestampInMilliseconds:timeStampInMilliseconds
                                                error:callbackError];
  });
}

@end
