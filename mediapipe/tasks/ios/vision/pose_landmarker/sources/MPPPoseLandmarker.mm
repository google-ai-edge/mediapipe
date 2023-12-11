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

#import "mediapipe/tasks/ios/vision/pose_landmarker/sources/MPPPoseLandmarker.h"

#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionPacketCreator.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/pose_landmarker/sources/MPPPoseLandmarksConnections.h"
#import "mediapipe/tasks/ios/vision/pose_landmarker/utils/sources/MPPPoseLandmarkerOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/pose_landmarker/utils/sources/MPPPoseLandmarkerResult+Helpers.h"

namespace {
using ::mediapipe::Timestamp;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // namespace

static NSString *const kImageTag = @"IMAGE";
static NSString *const kImageInStreamName = @"image_in";
static NSString *const kNormRectTag = @"NORM_RECT";
static NSString *const kNormRectInStreamName = @"norm_rect_in";
static NSString *const kImageOutStreamName = @"image_out";
static NSString *const kPoseLandmarksTag = @"NORM_LANDMARKS";
static NSString *const kPoseLandmarksOutStreamName = @"pose_landmarks";
static NSString *const kWorldLandmarksTag = @"WORLD_LANDMARKS";
static NSString *const kWorldLandmarksOutStreamName = @"world_landmarks";
static NSString *const kSegmentationMasksTag = @"SEGMENTATION_MASK";
static NSString *const kSegmentationMasksOutStreamName = @"segmentation_masks";
static NSString *const kTaskGraphName =
    @"mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph";
static NSString *const kTaskName = @"poseLandmarker";

#define InputPacketMap(imagePacket, normalizedRectPacket)   \
  {                                                         \
    {kImageInStreamName.cppString, imagePacket}, {          \
      kNormRectInStreamName.cppString, normalizedRectPacket \
    }                                                       \
  }

#define PoseLandmarkerResultWithOutputPacketMap(outputPacketMap)                                \
  ([MPPPoseLandmarkerResult                                                                     \
      poseLandmarkerResultWithLandmarksPacket:outputPacketMap[kPoseLandmarksOutStreamName       \
                                                                  .cppString]                   \
                         worldLandmarksPacket:outputPacketMap[kWorldLandmarksOutStreamName      \
                                                                  .cppString]                   \
                      segmentationMasksPacket:&(outputPacketMap[kSegmentationMasksOutStreamName \
                                                                    .cppString])])

@interface MPPPoseLandmarker () {
  /** iOS Vision Task Runner */
  MPPVisionTaskRunner *_visionTaskRunner;
  dispatch_queue_t _callbackQueue;
}
@property(nonatomic, weak) id<MPPPoseLandmarkerLiveStreamDelegate> poseLandmarkerLiveStreamDelegate;
@end

@implementation MPPPoseLandmarker

#pragma mark - Public

- (instancetype)initWithOptions:(MPPPoseLandmarkerOptions *)options error:(NSError **)error {
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
                      stringWithFormat:@"%@:%@", kPoseLandmarksTag, kPoseLandmarksOutStreamName],
                  [NSString
                      stringWithFormat:@"%@:%@", kWorldLandmarksTag, kWorldLandmarksOutStreamName],
                  [NSString stringWithFormat:@"%@:%@", kSegmentationMasksTag,
                                             kSegmentationMasksOutStreamName],
                  [NSString stringWithFormat:@"%@:%@", kImageTag, kImageOutStreamName]
                ]
                  taskOptions:options
           enableFlowLimiting:options.runningMode == MPPRunningModeLiveStream
                        error:error];

    if (!taskInfo) {
      return nil;
    }

    PacketsCallback packetsCallback = nullptr;

    if (options.poseLandmarkerLiveStreamDelegate) {
      _poseLandmarkerLiveStreamDelegate = options.poseLandmarkerLiveStreamDelegate;

      // Create a private serial dispatch queue in which the deleagte method will be called
      // asynchronously. This is to ensure that if the client performs a long running operation in
      // the delegate method, the queue on which the C++ callbacks is invoked is not blocked and is
      // freed up to continue with its operations.
      _callbackQueue = dispatch_queue_create(
          [MPPVisionTaskRunner uniqueDispatchQueueNameWithSuffix:kTaskName], nullptr);

      // Capturing `self` as weak in order to avoid `self` being kept in memory
      // and cause a retain cycle, after self is set to `nil`.
      MPPPoseLandmarker *__weak weakSelf = self;
      packetsCallback = [=](absl::StatusOr<PacketMap> liveStreamResult) {
        [weakSelf processLiveStreamResult:liveStreamResult];
      };
    }

    _visionTaskRunner = [[MPPVisionTaskRunner alloc] initWithTaskInfo:taskInfo
                                                          runningMode:options.runningMode
                                                           roiAllowed:NO
                                                      packetsCallback:std::move(packetsCallback)
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
  MPPPoseLandmarkerOptions *options = [[MPPPoseLandmarkerOptions alloc] init];

  options.baseOptions.modelAssetPath = modelPath;

  return [self initWithOptions:options error:error];
}

- (nullable MPPPoseLandmarkerResult *)detectImage:(MPPImage *)image error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap = [_visionTaskRunner processImage:image error:error];

  return [MPPPoseLandmarker poseLandmarkerResultWithOptionalOutputPacketMap:outputPacketMap];
}

- (nullable MPPPoseLandmarkerResult *)detectVideoFrame:(MPPImage *)image
                               timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                 error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap =
      [_visionTaskRunner processVideoFrame:image
                   timestampInMilliseconds:timestampInMilliseconds
                                     error:error];

  return [MPPPoseLandmarker poseLandmarkerResultWithOptionalOutputPacketMap:outputPacketMap];
}

- (BOOL)detectAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  return [_visionTaskRunner processLiveStreamImage:image
                           timestampInMilliseconds:timestampInMilliseconds
                                             error:error];
}

+ (NSArray<MPPConnection *> *)poseLandmarks {
  return MPPPoseLandmarksConnections;
}

#pragma mark - Private

- (void)processLiveStreamResult:(absl::StatusOr<PacketMap>)liveStreamResult {
  if (![self.poseLandmarkerLiveStreamDelegate
          respondsToSelector:@selector(poseLandmarker:
                                 didFinishDetectionWithResult:timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  if (![MPPCommonUtils checkCppError:liveStreamResult.status() toError:&callbackError]) {
    dispatch_async(_callbackQueue, ^{
      [self.poseLandmarkerLiveStreamDelegate poseLandmarker:self
                               didFinishDetectionWithResult:nil
                                    timestampInMilliseconds:Timestamp::Unset().Value()
                                                      error:callbackError];
    });
    return;
  }

  PacketMap &outputPacketMap = liveStreamResult.value();
  if (outputPacketMap[kImageOutStreamName.cppString].IsEmpty()) {
    return;
  }

  MPPPoseLandmarkerResult *result = PoseLandmarkerResultWithOutputPacketMap(outputPacketMap);

  NSInteger timestampInMilliseconds =
      outputPacketMap[kImageOutStreamName.cppString].Timestamp().Value() /
      kMicrosecondsPerMillisecond;
  dispatch_async(_callbackQueue, ^{
    [self.poseLandmarkerLiveStreamDelegate poseLandmarker:self
                             didFinishDetectionWithResult:result
                                  timestampInMilliseconds:timestampInMilliseconds
                                                    error:callbackError];
  });
}

+ (nullable MPPPoseLandmarkerResult *)poseLandmarkerResultWithOptionalOutputPacketMap:
    (std::optional<PacketMap> &)outputPacketMap {
  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return PoseLandmarkerResultWithOutputPacketMap(outputPacketMap.value());
}

@end
