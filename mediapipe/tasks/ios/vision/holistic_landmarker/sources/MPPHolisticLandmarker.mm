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

#import "mediapipe/tasks/ios/vision/holistic_landmarker/sources/MPPHolisticLandmarker.h"

#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionPacketCreator.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/holistic_landmarker/utils/sources/MPPHolisticLandmarkerOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/holistic_landmarker/utils/sources/MPPHolisticLandmarkerResult+Helpers.h"

namespace {
static const int kMicrosecondsPerMillisecond = 1000;

using ::mediapipe::Timestamp;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // namespace

static NSString *const kImageTag = @"IMAGE";
static NSString *const kImageInStreamName = @"image_in";
static NSString *const kImageOutStreamName = @"image_out";
static NSString *const kFaceLandmarksTag = @"FACE_LANDMARKS";
static NSString *const kFaceLandmarksOutStreamName = @"face_landmarks";
static NSString *const kFaceBlendshapesTag = @"FACE_BLENDSHAPES";
static NSString *const kFaceBlendshapesOutStreamName = @"face_blendshapes";
static NSString *const kPoseLandmarksTag = @"POSE_LANDMARKS";
static NSString *const kPoseLandmarksOutStreamName = @"pose_landmarks";
static NSString *const kPoseWorldLandmarksTag = @"POSE_WORLD_LANDMARKS";
static NSString *const kPoseWorldLandmarksOutStreamName = @"pose_world_landmarks";
static NSString *const kLeftHandLandmarksTag = @"LEFT_HAND_LANDMARKS";
static NSString *const kLeftHandLandmarksOutStreamName = @"left_hand_landmarks";
static NSString *const kLeftHandWorldLandmarksTag = @"LEFT_HAND_WORLD_LANDMARKS";
static NSString *const kLeftHandWorldLandmarksOutStreamName = @"left_hand_world_landmarks";
static NSString *const kRightHandLandmarksTag = @"RIGHT_HAND_LANDMARKS";
static NSString *const kRightHandLandmarksOutStreamName = @"right_hand_landmarks";
static NSString *const kRightHandWorldLandmarksTag = @"RIGHT_HAND_WORLD_LANDMARKS";
static NSString *const kRightHandWorldLandmarksOutStreamName = @"right_hand_world_landmarks";
static NSString *const kPoseSegmentationMaskTag = @"POSE_SEGMENTATION_MASK";
static NSString *const kPoseSegmentationMaskOutStreamName = @"pose_segmentation_mask";
static NSString *const kTaskGraphName =
    @"mediapipe.tasks.vision.holistic_landmarker.HolisticLandmarkerGraph";
static NSString *const kTaskName = @"holisticLandmarker";

#define HolisticLandmarkerResultWithOutputPacketMap(outputPacketMap)                              \
  [MPPHolisticLandmarkerResult                                                                    \
      holisticLandmarkerResultWithFaceLandmarksPacket:outputPacketMap[kFaceLandmarksOutStreamName \
                                                                          .cppString]             \
                                faceBlendshapesPacket:                                            \
                                    outputPacketMap[kFaceBlendshapesOutStreamName.cppString]      \
                                  poseLandmarksPacket:outputPacketMap[kPoseLandmarksOutStreamName \
                                                                          .cppString]             \
                             poseWorldLandmarksPacket:                                            \
                                 outputPacketMap[kPoseWorldLandmarksOutStreamName.cppString]      \
                           poseSegmentationMaskPacket:                                            \
                               outputPacketMap[kPoseSegmentationMaskOutStreamName.cppString]      \
                              leftHandLandmarksPacket:                                            \
                                  outputPacketMap[kLeftHandLandmarksOutStreamName.cppString]      \
                         leftHandWorldLandmarksPacket:                                            \
                             outputPacketMap[kLeftHandWorldLandmarksOutStreamName.cppString]      \
                             rightHandLandmarksPacket:                                            \
                                 outputPacketMap[kRightHandLandmarksOutStreamName.cppString]      \
                        rightHandWorldLandmarksPacket:                                            \
                            outputPacketMap[kRightHandWorldLandmarksOutStreamName.cppString]]

@interface MPPHolisticLandmarker () {
  /** iOS Vision Task Runner */
  MPPVisionTaskRunner *_visionTaskRunner;
  dispatch_queue_t _callbackQueue;
}
@property(nonatomic, weak) id<MPPHolisticLandmarkerLiveStreamDelegate>
    holisticLandmarkerLiveStreamDelegate;
@end

@implementation MPPHolisticLandmarker

#pragma mark - Public

- (instancetype)initWithOptions:(MPPHolisticLandmarkerOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    NSMutableArray<NSString *> *outputStreams = [NSMutableArray
        arrayWithObjects:[NSString stringWithFormat:@"%@:%@", kFaceLandmarksTag,
                                                    kFaceLandmarksOutStreamName],
                         [NSString stringWithFormat:@"%@:%@", kPoseLandmarksTag,
                                                    kPoseLandmarksOutStreamName],
                         [NSString stringWithFormat:@"%@:%@", kPoseWorldLandmarksTag,
                                                    kPoseWorldLandmarksOutStreamName],
                         [NSString stringWithFormat:@"%@:%@", kLeftHandLandmarksTag,
                                                    kLeftHandLandmarksOutStreamName],
                         [NSString stringWithFormat:@"%@:%@", kLeftHandWorldLandmarksTag,
                                                    kLeftHandWorldLandmarksOutStreamName],
                         [NSString stringWithFormat:@"%@:%@", kRightHandLandmarksTag,
                                                    kRightHandLandmarksOutStreamName],
                         [NSString stringWithFormat:@"%@:%@", kRightHandWorldLandmarksTag,
                                                    kRightHandWorldLandmarksOutStreamName],
                         [NSString stringWithFormat:@"%@:%@", kImageTag, kImageOutStreamName], nil];

    if (options.outputPoseSegmentationMasks) {
      [outputStreams addObject:[NSString stringWithFormat:@"%@:%@", kPoseSegmentationMaskTag,
                                                          kPoseSegmentationMaskOutStreamName]];
    }
    if (options.outputFaceBlendshapes) {
      [outputStreams addObject:[NSString stringWithFormat:@"%@:%@", kFaceBlendshapesTag,
                                                          kFaceBlendshapesOutStreamName]];
    }

    MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
        initWithTaskGraphName:kTaskGraphName
                 inputStreams:@[
                   [NSString stringWithFormat:@"%@:%@", kImageTag, kImageInStreamName],
                 ]
                outputStreams:outputStreams
                  taskOptions:options
           enableFlowLimiting:options.runningMode == MPPRunningModeLiveStream
                        error:error];

    if (!taskInfo) {
      return nil;
    }

    PacketsCallback packetsCallback = nullptr;

    if (options.holisticLandmarkerLiveStreamDelegate) {
      _holisticLandmarkerLiveStreamDelegate = options.holisticLandmarkerLiveStreamDelegate;

      // Create a private serial dispatch queue in which the deleagte method will be called
      // asynchronously. This is to ensure that if the client performs a long running operation in
      // the delegate method, the queue on which the C++ callbacks is invoked is not blocked and is
      // freed up to continue with its operations.
      _callbackQueue = dispatch_queue_create(
          [MPPVisionTaskRunner uniqueDispatchQueueNameWithSuffix:kTaskName], nullptr);

      // Capturing `self` as weak in order to avoid `self` being kept in memory
      // and cause a retain cycle, after self is set to `nil`.
      MPPHolisticLandmarker *__weak weakSelf = self;
      packetsCallback = [=](absl::StatusOr<PacketMap> liveStreamResult) {
        [weakSelf processLiveStreamResult:liveStreamResult];
      };
    }

    _visionTaskRunner = [[MPPVisionTaskRunner alloc] initWithTaskInfo:taskInfo
                                                          runningMode:options.runningMode
                                                           roiAllowed:NO
                                                      packetsCallback:std::move(packetsCallback)
                                                 imageInputStreamName:kImageInStreamName
                                              normRectInputStreamName:nil
                                                                error:error];

    if (!_visionTaskRunner) {
      return nil;
    }
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPHolisticLandmarkerOptions *options = [[MPPHolisticLandmarkerOptions alloc] init];

  options.baseOptions.modelAssetPath = modelPath;

  return [self initWithOptions:options error:error];
}

- (nullable MPPHolisticLandmarkerResult *)detectImage:(MPPImage *)image error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap = [_visionTaskRunner processImage:image error:error];

  return
      [MPPHolisticLandmarker holisticLandmarkerResultWithOptionalOutputPacketMap:outputPacketMap];
}

- (nullable MPPHolisticLandmarkerResult *)detectVideoFrame:(MPPImage *)image
                                   timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                     error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap =
      [_visionTaskRunner processVideoFrame:image
                   timestampInMilliseconds:timestampInMilliseconds
                                     error:error];

  return
      [MPPHolisticLandmarker holisticLandmarkerResultWithOptionalOutputPacketMap:outputPacketMap];
}

- (BOOL)detectAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  return [_visionTaskRunner processLiveStreamImage:image
                           timestampInMilliseconds:timestampInMilliseconds
                                             error:error];
}

#pragma mark - Private

- (void)processLiveStreamResult:(absl::StatusOr<PacketMap>)liveStreamResult {
  if (![self.holisticLandmarkerLiveStreamDelegate
          respondsToSelector:@selector(holisticLandmarker:
                                 didFinishDetectionWithResult:timestampInMilliseconds:error:)]) {
    return;
  }

  NSError *callbackError = nil;
  if (![MPPCommonUtils checkCppError:liveStreamResult.status() toError:&callbackError]) {
    dispatch_async(_callbackQueue, ^{
      [self.holisticLandmarkerLiveStreamDelegate holisticLandmarker:self
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

  MPPHolisticLandmarkerResult *result =
      HolisticLandmarkerResultWithOutputPacketMap(outputPacketMap);

  NSInteger timestampInMilliseconds =
      outputPacketMap[kImageOutStreamName.cppString].Timestamp().Value() /
      kMicrosecondsPerMillisecond;
  dispatch_async(_callbackQueue, ^{
    [self.holisticLandmarkerLiveStreamDelegate holisticLandmarker:self
                                     didFinishDetectionWithResult:result
                                          timestampInMilliseconds:timestampInMilliseconds
                                                            error:callbackError];
  });
}

+ (nullable MPPHolisticLandmarkerResult *)holisticLandmarkerResultWithOptionalOutputPacketMap:
    (std::optional<PacketMap> &)outputPacketMap {
  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return HolisticLandmarkerResultWithOutputPacketMap(outputPacketMap.value());
}

@end
