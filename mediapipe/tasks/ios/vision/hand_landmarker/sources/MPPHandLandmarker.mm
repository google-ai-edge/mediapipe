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

#import "mediapipe/tasks/ios/vision/hand_landmarker/sources/MPPHandLandmarker.h"

#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionPacketCreator.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/hand_landmarker/sources/MPPHandLandmarksConnections.h"
#import "mediapipe/tasks/ios/vision/hand_landmarker/utils/sources/MPPHandLandmarkerOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/hand_landmarker/utils/sources/MPPHandLandmarkerResult+Helpers.h"

namespace {
using ::mediapipe::NormalizedRect;
using ::mediapipe::Packet;
using ::mediapipe::Timestamp;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // namespace

static NSString *const kImageTag = @"IMAGE";
static NSString *const kImageInStreamName = @"image_in";
static NSString *const kNormRectTag = @"NORM_RECT";
static NSString *const kNormRectInStreamName = @"norm_rect_in";
static NSString *const kImageOutStreamName = @"image_out";
static NSString *const kLandmarksTag = @"LANDMARKS";
static NSString *const kLandmarksOutStreamName = @"hand_landmarks";
static NSString *const kWorldLandmarksTag = @"WORLD_LANDMARKS";
static NSString *const kWorldLandmarksOutStreamName = @"world_hand_landmarks";
static NSString *const kHandednessTag = @"HANDEDNESS";
static NSString *const kHandednessOutStreamName = @"handedness";
static NSString *const kTaskGraphName =
    @"mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph";
static NSString *const kTaskName = @"handLandmarker";

#define InputPacketMap(imagePacket, normalizedRectPacket)   \
  {                                                         \
    {kImageInStreamName.cppString, imagePacket}, {          \
      kNormRectInStreamName.cppString, normalizedRectPacket \
    }                                                       \
  }

#define HandLandmarkerResultWithOutputPacketMap(outputPacketMap)                                 \
  ([MPPHandLandmarkerResult                                                                      \
      handLandmarkerResultWithLandmarksPacket:outputPacketMap[kLandmarksOutStreamName.cppString] \
                         worldLandmarksPacket:outputPacketMap[kWorldLandmarksOutStreamName       \
                                                                  .cppString]                    \
                             handednessPacket:outputPacketMap[kHandednessOutStreamName           \
                                                                  .cppString]])

@interface MPPHandLandmarker () {
  /** iOS Vision Task Runner */
  MPPVisionTaskRunner *_visionTaskRunner;
  dispatch_queue_t _callbackQueue;
}
@property(nonatomic, weak) id<MPPHandLandmarkerLiveStreamDelegate> handLandmarkerLiveStreamDelegate;
@end

@implementation MPPHandLandmarker

#pragma mark - Public

- (instancetype)initWithOptions:(MPPHandLandmarkerOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
        initWithTaskGraphName:kTaskGraphName
                 inputStreams:@[
                   [NSString stringWithFormat:@"%@:%@", kImageTag, kImageInStreamName],
                   [NSString stringWithFormat:@"%@:%@", kNormRectTag, kNormRectInStreamName]
                 ]
                outputStreams:@[
                  [NSString stringWithFormat:@"%@:%@", kLandmarksTag, kLandmarksOutStreamName],
                  [NSString
                      stringWithFormat:@"%@:%@", kWorldLandmarksTag, kWorldLandmarksOutStreamName],
                  [NSString stringWithFormat:@"%@:%@", kHandednessTag, kHandednessOutStreamName],
                  [NSString stringWithFormat:@"%@:%@", kImageTag, kImageOutStreamName]
                ]
                  taskOptions:options
           enableFlowLimiting:options.runningMode == MPPRunningModeLiveStream
                        error:error];

    if (!taskInfo) {
      return nil;
    }

    PacketsCallback packetsCallback = nullptr;

    if (options.handLandmarkerLiveStreamDelegate) {
      _handLandmarkerLiveStreamDelegate = options.handLandmarkerLiveStreamDelegate;

      // Create a private serial dispatch queue in which the deleagte method will be called
      // asynchronously. This is to ensure that if the client performs a long running operation in
      // the delegate method, the queue on which the C++ callbacks is invoked is not blocked and is
      // freed up to continue with its operations.
      _callbackQueue = dispatch_queue_create(
          [MPPVisionTaskRunner uniqueDispatchQueueNameWithSuffix:kTaskName], NULL);

      // Capturing `self` as weak in order to avoid `self` being kept in memory
      // and cause a retain cycle, after self is set to `nil`.
      MPPHandLandmarker *__weak weakSelf = self;
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
  MPPHandLandmarkerOptions *options = [[MPPHandLandmarkerOptions alloc] init];

  options.baseOptions.modelAssetPath = modelPath;

  return [self initWithOptions:options error:error];
}

- (nullable MPPHandLandmarkerResult *)detectImage:(MPPImage *)image error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap = [_visionTaskRunner processImage:image error:error];

  return [MPPHandLandmarker handLandmarkerResultWithOptionalOutputPacketMap:outputPacketMap];
}

- (nullable MPPHandLandmarkerResult *)detectVideoFrame:(MPPImage *)image
                                 timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                   error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap =
      [_visionTaskRunner processVideoFrame:image
                   timestampInMilliseconds:timestampInMilliseconds
                                     error:error];

  return [MPPHandLandmarker handLandmarkerResultWithOptionalOutputPacketMap:outputPacketMap];
}

- (BOOL)detectAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  return [_visionTaskRunner processLiveStreamImage:image
                           timestampInMilliseconds:timestampInMilliseconds
                                             error:error];
}

+ (NSArray<MPPConnection *> *)handPalmConnections {
  return MPPHandPalmConnections;
}

+ (NSArray<MPPConnection *> *)handThumbConnections {
  return MPPHandThumbConnections;
}

+ (NSArray<MPPConnection *> *)handIndexFingerConnections {
  return MPPHandIndexFingerConnections;
}

+ (NSArray<MPPConnection *> *)handMiddleFingerConnections {
  return MPPHandMiddleFingerConnections;
}

+ (NSArray<MPPConnection *> *)handRingFingerConnections {
  return MPPHandRingFingerConnections;
}

+ (NSArray<MPPConnection *> *)handPinkyConnections {
  return MPPHandPinkyConnections;
}

+ (NSArray<MPPConnection *> *)handConnections {
  return MPPHandConnections;
}

#pragma mark - Private

- (void)processLiveStreamResult:(absl::StatusOr<PacketMap>)liveStreamResult {
  if (![self.handLandmarkerLiveStreamDelegate
          respondsToSelector:@selector(handLandmarker:
                                 didFinishDetectionWithResult:timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  if (![MPPCommonUtils checkCppError:liveStreamResult.status() toError:&callbackError]) {
    dispatch_async(_callbackQueue, ^{
      [self.handLandmarkerLiveStreamDelegate handLandmarker:self
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

  MPPHandLandmarkerResult *result = HandLandmarkerResultWithOutputPacketMap(outputPacketMap);

  NSInteger timestampInMilliseconds =
      outputPacketMap[kImageOutStreamName.cppString].Timestamp().Value() /
      kMicrosecondsPerMillisecond;
  dispatch_async(_callbackQueue, ^{
    [self.handLandmarkerLiveStreamDelegate handLandmarker:self
                             didFinishDetectionWithResult:result
                                  timestampInMilliseconds:timestampInMilliseconds
                                                    error:callbackError];
  });
}

+ (nullable MPPHandLandmarkerResult *)handLandmarkerResultWithOptionalOutputPacketMap:
    (std::optional<PacketMap> &)outputPacketMap {
  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return HandLandmarkerResultWithOutputPacketMap(outputPacketMap.value());
}

@end
