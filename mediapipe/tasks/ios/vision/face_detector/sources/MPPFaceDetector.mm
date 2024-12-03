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

#import "mediapipe/tasks/ios/vision/face_detector/sources/MPPFaceDetector.h"

#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionPacketCreator.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/face_detector/utils/sources/MPPFaceDetectorOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/face_detector/utils/sources/MPPFaceDetectorResult+Helpers.h"

using ::mediapipe::Timestamp;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;

static constexpr int kMicrosecondsPerMillisecond = 1000;

// Constants for the underlying MP Tasks Graph. See
// https://github.com/google/mediapipe/tree/master/mediapipe/tasks/cc/vision/face_detector/face_detector_graph.cc
static NSString *const kDetectionsStreamName = @"detections_out";
static NSString *const kDetectionsTag = @"DETECTIONS";
static NSString *const kImageInStreamName = @"image_in";
static NSString *const kImageOutStreamName = @"image_out";
static NSString *const kImageTag = @"IMAGE";
static NSString *const kNormRectStreamName = @"norm_rect_in";
static NSString *const kNormRectTag = @"NORM_RECT";
static NSString *const kTaskGraphName = @"mediapipe.tasks.vision.face_detector.FaceDetectorGraph";
static NSString *const kTaskName = @"faceDetector";

#define InputPacketMap(imagePacket, normalizedRectPacket) \
  {                                                       \
    {kImageInStreamName.cppString, imagePacket}, {        \
      kNormRectStreamName.cppString, normalizedRectPacket \
    }                                                     \
  }

#define FaceDetectorResultWithOutputPacketMap(outputPacketMap)                                   \
  (                                                                                              \
    [MPPFaceDetectorResult                                                                       \
        faceDetectorResultWithDetectionsPacket:outputPacketMap[kDetectionsStreamName.cppString]] \
  )

@interface MPPFaceDetector () {
  /** iOS Vision Task Runner */
  MPPVisionTaskRunner *_visionTaskRunner;
  dispatch_queue_t _callbackQueue;
}
@property(nonatomic, weak) id<MPPFaceDetectorLiveStreamDelegate> faceDetectorLiveStreamDelegate;

- (void)processLiveStreamResult:(absl::StatusOr<PacketMap>)liveStreamResult;
@end

@implementation MPPFaceDetector

- (instancetype)initWithOptions:(MPPFaceDetectorOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
        initWithTaskGraphName:kTaskGraphName
                 inputStreams:@[
                   [NSString stringWithFormat:@"%@:%@", kImageTag, kImageInStreamName],
                   [NSString stringWithFormat:@"%@:%@", kNormRectTag, kNormRectStreamName]
                 ]
                outputStreams:@[
                  [NSString stringWithFormat:@"%@:%@", kDetectionsTag, kDetectionsStreamName],
                  [NSString stringWithFormat:@"%@:%@", kImageTag, kImageOutStreamName]
                ]
                  taskOptions:options
           enableFlowLimiting:options.runningMode == MPPRunningModeLiveStream
                        error:error];

    if (!taskInfo) {
      return nil;
    }

    PacketsCallback packetsCallback = nullptr;

    if (options.faceDetectorLiveStreamDelegate) {
      _faceDetectorLiveStreamDelegate = options.faceDetectorLiveStreamDelegate;

      // Create a private serial dispatch queue in which the delegate method will be called
      // asynchronously. This is to ensure that if the client performs a long running operation in
      // the delegate method, the queue on which the C++ callbacks is invoked is not blocked and is
      // freed up to continue with its operations.
      _callbackQueue = dispatch_queue_create(
          [MPPVisionTaskRunner uniqueDispatchQueueNameWithSuffix:kTaskName], NULL);

      // Capturing `self` as weak in order to avoid `self` being kept in memory
      // and cause a retain cycle, after self is set to `nil`.
      MPPFaceDetector *__weak weakSelf = self;
      packetsCallback = [=](absl::StatusOr<PacketMap> liveStreamResult) {
        [weakSelf processLiveStreamResult:liveStreamResult];
      };
    }

    _visionTaskRunner = [[MPPVisionTaskRunner alloc] initWithTaskInfo:taskInfo
                                                          runningMode:options.runningMode
                                                           roiAllowed:NO
                                                      packetsCallback:std::move(packetsCallback)
                                                 imageInputStreamName:kImageInStreamName
                                              normRectInputStreamName:kNormRectStreamName
                                                                error:error];

    if (!_visionTaskRunner) {
      return nil;
    }
  }

  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPFaceDetectorOptions *options = [[MPPFaceDetectorOptions alloc] init];

  options.baseOptions.modelAssetPath = modelPath;

  return [self initWithOptions:options error:error];
}

- (nullable MPPFaceDetectorResult *)detectImage:(MPPImage *)image error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap = [_visionTaskRunner processImage:image error:error];

  return [MPPFaceDetector faceDetectorResultWithOptionalOutputPacketMap:outputPacketMap];
}

- (nullable MPPFaceDetectorResult *)detectVideoFrame:(MPPImage *)image
                               timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                 error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap =
      [_visionTaskRunner processVideoFrame:image
                   timestampInMilliseconds:timestampInMilliseconds
                                     error:error];

  return [MPPFaceDetector faceDetectorResultWithOptionalOutputPacketMap:outputPacketMap];
}

- (BOOL)detectAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  return [_visionTaskRunner processLiveStreamImage:image
                           timestampInMilliseconds:timestampInMilliseconds
                                             error:error];
}

- (void)processLiveStreamResult:(absl::StatusOr<PacketMap>)liveStreamResult {
  if (![self.faceDetectorLiveStreamDelegate
          respondsToSelector:@selector(faceDetector:
                                 didFinishDetectionWithResult:timestampInMilliseconds:error:)]) {
    return;
  }
  NSError *callbackError = nil;
  if (![MPPCommonUtils checkCppError:liveStreamResult.status() toError:&callbackError]) {
    dispatch_async(_callbackQueue, ^{
      [self.faceDetectorLiveStreamDelegate faceDetector:self
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

  MPPFaceDetectorResult *result = FaceDetectorResultWithOutputPacketMap(liveStreamResult.value());

  NSInteger timestampInMilliseconds =
      outputPacketMap[kImageOutStreamName.cppString].Timestamp().Value() /
      kMicrosecondsPerMillisecond;
  dispatch_async(_callbackQueue, ^{
    [self.faceDetectorLiveStreamDelegate faceDetector:self
                         didFinishDetectionWithResult:result
                              timestampInMilliseconds:timestampInMilliseconds
                                                error:callbackError];
  });
}

+ (nullable MPPFaceDetectorResult *)faceDetectorResultWithOptionalOutputPacketMap:
    (std::optional<PacketMap>)outputPacketMap {
  if (!outputPacketMap.has_value()) {
    return nil;
  }

  return FaceDetectorResultWithOutputPacketMap(outputPacketMap.value());
}

@end
