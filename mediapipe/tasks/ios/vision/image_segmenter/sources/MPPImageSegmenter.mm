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

#import "mediapipe/tasks/ios/vision/image_segmenter/sources/MPPImageSegmenter.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/image_segmenter/utils/sources/MPPImageSegmenterOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/image_segmenter/utils/sources/MPPImageSegmenterResult+Helpers.h"

#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/util/label_map.pb.h"

static constexpr int kMicrosecondsPerMillisecond = 1000;

// Constants for the underlying MP Tasks Graph. See
// https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/tasks/cc/vision/image_segmenter/image_segmenter_graph.cc
static NSString *const kConfidenceMasksStreamName = @"confidence_masks";
static NSString *const kConfidenceMasksTag = @"CONFIDENCE_MASKS";
static NSString *const kCategoryMaskStreamName = @"category_mask";
static NSString *const kCategoryMaskTag = @"CATEGORY_MASK";
static NSString *const kQualityScoresStreamName = @"quality_scores";
static NSString *const kQualityScoresTag = @"QUALITY_SCORES";
static NSString *const kImageInStreamName = @"image_in";
static NSString *const kImageOutStreamName = @"image_out";
static NSString *const kImageTag = @"IMAGE";
static NSString *const kNormRectStreamName = @"norm_rect_in";
static NSString *const kNormRectTag = @"NORM_RECT";
static NSString *const kTaskGraphName =
    @"mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph";
static NSString *const kTaskName = @"imageSegmenter";

#define InputPacketMap(imagePacket, normalizedRectPacket) \
  {                                                       \
    {kImageInStreamName.cppString, imagePacket}, {        \
      kNormRectStreamName.cppString, normalizedRectPacket \
    }                                                     \
  }

namespace {
using ::mediapipe::CalculatorGraphConfig;
using ::mediapipe::Timestamp;
using ::mediapipe::tasks::TensorsToSegmentationCalculatorOptions;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // anonymous namespace

@interface MPPImageSegmenter () {
  /** iOS Vision Task Runner */
  MPPVisionTaskRunner *_visionTaskRunner;
  dispatch_queue_t _callbackQueue;
}
@property(nonatomic, weak) id<MPPImageSegmenterLiveStreamDelegate> imageSegmenterLiveStreamDelegate;

- (void)processLiveStreamResult:(absl::StatusOr<PacketMap>)liveStreamResult;
@end

@implementation MPPImageSegmenter

#pragma mark - Public

- (instancetype)initWithOptions:(MPPImageSegmenterOptions *)options error:(NSError **)error {
  self = [super init];
  if (self) {
    NSMutableArray<NSString *> *outputStreams = [NSMutableArray
        arrayWithObjects:[NSString stringWithFormat:@"%@:%@", kQualityScoresTag,
                                                    kQualityScoresStreamName],
                         [NSString stringWithFormat:@"%@:%@", kImageTag, kImageOutStreamName], nil];
    if (options.shouldOutputConfidenceMasks) {
      [outputStreams addObject:[NSString stringWithFormat:@"%@:%@", kConfidenceMasksTag,
                                                          kConfidenceMasksStreamName]];
    }
    if (options.shouldOutputCategoryMask) {
      [outputStreams addObject:[NSString stringWithFormat:@"%@:%@", kCategoryMaskTag,
                                                          kCategoryMaskStreamName]];
    }

    MPPTaskInfo *taskInfo = [[MPPTaskInfo alloc]
        initWithTaskGraphName:kTaskGraphName
                 inputStreams:@[
                   [NSString stringWithFormat:@"%@:%@", kImageTag, kImageInStreamName],
                   [NSString stringWithFormat:@"%@:%@", kNormRectTag, kNormRectStreamName]
                 ]
                outputStreams:outputStreams
                  taskOptions:options
           enableFlowLimiting:options.runningMode == MPPRunningModeLiveStream
                        error:error];

    if (!taskInfo) {
      return nil;
    }

    PacketsCallback packetsCallback = nullptr;

    if (options.imageSegmenterLiveStreamDelegate) {
      _imageSegmenterLiveStreamDelegate = options.imageSegmenterLiveStreamDelegate;

      // Create a private serial dispatch queue in which the delegate method will be called
      // asynchronously. This is to ensure that if the client performs a long running operation in
      // the delegate method, the queue on which the C++ callbacks is invoked is not blocked and is
      // freed up to continue with its operations.
      _callbackQueue = dispatch_queue_create(
          [MPPVisionTaskRunner uniqueDispatchQueueNameWithSuffix:kTaskName], nullptr);

      // Capturing `self` as weak in order to avoid `self` being kept in memory
      // and cause a retain cycle, after self is set to `nil`.
      MPPImageSegmenter *__weak weakSelf = self;
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

    _labels = [MPPImageSegmenter populateLabelsWithGraphConfig:_visionTaskRunner.graphConfig
                                                         error:error];
    if (!_labels) {
      return nil;
    }
  }

  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPImageSegmenterOptions *options = [[MPPImageSegmenterOptions alloc] init];

  options.baseOptions.modelAssetPath = modelPath;

  return [self initWithOptions:options error:error];
}

- (nullable MPPImageSegmenterResult *)segmentImage:(MPPImage *)image error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap = [_visionTaskRunner processImage:image error:error];
  return [MPPImageSegmenter imageSegmenterResultWithOptionalOutputPacketMap:outputPacketMap
                                                   shouldCopyMaskPacketData:YES];
}

- (void)segmentImage:(MPPImage *)image
    withCompletionHandler:(void (^)(MPPImageSegmenterResult *_Nullable result,
                                    NSError *_Nullable error))completionHandler {
  NSError *error = nil;
  std::optional<PacketMap> outputPacketMap = [_visionTaskRunner processImage:image error:&error];

  MPPImageSegmenterResult *result =
      [MPPImageSegmenter imageSegmenterResultWithOptionalOutputPacketMap:outputPacketMap
                                                shouldCopyMaskPacketData:NO];
  completionHandler(result, error);
}

- (nullable MPPImageSegmenterResult *)segmentVideoFrame:(MPPImage *)image
                                timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                  error:(NSError **)error {
  std::optional<PacketMap> outputPacketMap =
      [_visionTaskRunner processVideoFrame:image
                   timestampInMilliseconds:timestampInMilliseconds
                                     error:error];

  return [MPPImageSegmenter imageSegmenterResultWithOptionalOutputPacketMap:outputPacketMap
                                                   shouldCopyMaskPacketData:YES];
}

- (void)segmentVideoFrame:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
      withCompletionHandler:(void (^)(MPPImageSegmenterResult *_Nullable result,
                                      NSError *_Nullable error))completionHandler {
  NSError *error = nil;
  std::optional<PacketMap> outputPacketMap =
      [_visionTaskRunner processVideoFrame:image
                   timestampInMilliseconds:timestampInMilliseconds
                                     error:&error];

  MPPImageSegmenterResult *result =
      [MPPImageSegmenter imageSegmenterResultWithOptionalOutputPacketMap:outputPacketMap
                                                shouldCopyMaskPacketData:NO];
  completionHandler(result, error);
}
- (BOOL)segmentAsyncImage:(MPPImage *)image
    timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                      error:(NSError **)error {
  return [_visionTaskRunner processLiveStreamImage:image
                           timestampInMilliseconds:timestampInMilliseconds
                                             error:error];
}

#pragma mark - Private

+ (NSArray<NSString *> *)populateLabelsWithGraphConfig:(const CalculatorGraphConfig &)graphConfig
                                                 error:(NSError **)error {
  bool found_tensor_to_segmentation_calculator = false;

  NSMutableArray<NSString *> *labels = [NSMutableArray arrayWithCapacity:(NSUInteger)graphConfig.node_size()];
  for (const auto &node : graphConfig.node()) {
    if (node.calculator() == "mediapipe.tasks.TensorsToSegmentationCalculator") {
      if (!found_tensor_to_segmentation_calculator) {
        found_tensor_to_segmentation_calculator = true;
      } else {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeFailedPreconditionError
                              description:@"The graph has more than one "
                                          @"`mediapipe.tasks.TensorsToSegmentationCalculator`."];
        return nil;
      }
      TensorsToSegmentationCalculatorOptions options =
          node.options().GetExtension(TensorsToSegmentationCalculatorOptions::ext);
      if (!options.label_items().empty()) {
        for (int i = 0; i < options.label_items_size(); ++i) {
          if (!options.label_items().contains(i)) {
            [MPPCommonUtils
                createCustomError:error
                         withCode:MPPTasksErrorCodeFailedPreconditionError
                      description:[NSString
                                      stringWithFormat:@"The lablemap has no expected key %d.", i]];

            return nil;
          }
          [labels addObject:[NSString stringWithCppString:options.label_items().at(i).name()]];
        }
      }
    }
  }
  return labels;
}

+ (nullable MPPImageSegmenterResult *)
    imageSegmenterResultWithOptionalOutputPacketMap:(std::optional<PacketMap> &)outputPacketMap
                           shouldCopyMaskPacketData:(BOOL)shouldCopyMaskPacketData {
  if (!outputPacketMap.has_value()) {
    return nil;
  }
  MPPImageSegmenterResult *result =
      [self imageSegmenterResultWithOutputPacketMap:outputPacketMap.value()
                           shouldCopyMaskPacketData:shouldCopyMaskPacketData];
  return result;
}

+ (nullable MPPImageSegmenterResult *)
    imageSegmenterResultWithOutputPacketMap:(PacketMap &)outputPacketMap
                   shouldCopyMaskPacketData:(BOOL)shouldCopyMaskPacketData {
  return [MPPImageSegmenterResult
      imageSegmenterResultWithConfidenceMasksPacket:outputPacketMap[kConfidenceMasksStreamName
                                                                        .cppString]
                                 categoryMaskPacket:outputPacketMap[kCategoryMaskStreamName
                                                                        .cppString]
                                qualityScoresPacket:outputPacketMap[kQualityScoresStreamName
                                                                        .cppString]
                            timestampInMilliseconds:outputPacketMap[kImageOutStreamName.cppString]
                                                        .Timestamp()
                                                        .Value() /
                                                    kMicrosecondsPerMillisecond
                           shouldCopyMaskPacketData:shouldCopyMaskPacketData];
}

- (void)processLiveStreamResult:(absl::StatusOr<PacketMap>)liveStreamResult {
  if (![self.imageSegmenterLiveStreamDelegate
          respondsToSelector:@selector(imageSegmenter:
                                 didFinishSegmentationWithResult:timestampInMilliseconds:error:)]) {
    return;
  }
  NSError *callbackError = nil;
  if (![MPPCommonUtils checkCppError:liveStreamResult.status() toError:&callbackError]) {
    dispatch_async(_callbackQueue, ^{
      [self.imageSegmenterLiveStreamDelegate imageSegmenter:self
                            didFinishSegmentationWithResult:nil
                                    timestampInMilliseconds:Timestamp::Unset().Value()
                                                      error:callbackError];
    });
    return;
  }

  // Output packet map is moved to a block variable that will not be deallocated for the lifetime of
  // the `dispatch_async` call. Since masks are not copied, this ensures that they are only
  // deallocated after the delegate call completes.
  __block PacketMap outputPacketMap = std::move(liveStreamResult.value());
  if (outputPacketMap[kImageOutStreamName.cppString].IsEmpty()) {
    return;
  }

  dispatch_async(_callbackQueue, ^{
    MPPImageSegmenterResult *result =
        [MPPImageSegmenter imageSegmenterResultWithOutputPacketMap:outputPacketMap
                                          shouldCopyMaskPacketData:NO];

    [self.imageSegmenterLiveStreamDelegate imageSegmenter:self
                          didFinishSegmentationWithResult:result
                                  timestampInMilliseconds:result.timestampInMilliseconds
                                                    error:callbackError];
  });
}

@end
