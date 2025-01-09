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

#import "mediapipe/tasks/ios/vision/interactive_segmenter/sources/MPPInteractiveSegmenter.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionPacketCreator.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/interactive_segmenter/utils/sources/MPPInteractiveSegmenterOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/interactive_segmenter/utils/sources/MPPInteractiveSegmenterResult+Helpers.h"

#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/util/label_map.pb.h"

static constexpr int kMicrosecondsPerMillisecond = 1000;

// Constants for the underlying MP Tasks Graph. See
// https://github.com/google/mediapipe/tree/master/mediapipe/tasks/cc/vision/image_segmenter/image_segmenter_graph.cc
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
static NSString *const kRoiInStreamName = @"roi_in";
static NSString *const kRoiTag = @"ROI";
static NSString *const kTaskGraphName =
    @"mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraph";
static NSString *const kTaskName = @"interactiveSegmenter";

#define InputPacketMap(imagePacket, normalizedRectPacket) \
  {                                                       \
    {kImageInStreamName.cppString, imagePacket}, {        \
      kNormRectStreamName.cppString, normalizedRectPacket \
    }                                                     \
  }

namespace {
using ::mediapipe::CalculatorGraphConfig;
using ::mediapipe::Packet;
using ::mediapipe::tasks::TensorsToSegmentationCalculatorOptions;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // anonymous namespace

@interface MPPInteractiveSegmenter () {
  /** iOS Vision Task Runner */
  MPPVisionTaskRunner *_visionTaskRunner;
}

@end

@implementation MPPInteractiveSegmenter

#pragma mark - Public

- (instancetype)initWithOptions:(MPPInteractiveSegmenterOptions *)options error:(NSError **)error {
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
                   [NSString stringWithFormat:@"%@:%@", kNormRectTag, kNormRectStreamName],
                   [NSString stringWithFormat:@"%@:%@", kRoiTag, kRoiInStreamName],
                 ]
                outputStreams:outputStreams
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
                                                      packetsCallback:std::move(packetsCallback)
                                                 imageInputStreamName:kImageInStreamName
                                              normRectInputStreamName:kNormRectStreamName
                                                                error:error];
    if (!_visionTaskRunner) {
      return nil;
    }

    _labels = [MPPInteractiveSegmenter populateLabelsWithGraphConfig:_visionTaskRunner.graphConfig
                                                               error:error];
    if (!_labels) {
      return nil;
    }
  }

  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPInteractiveSegmenterOptions *options = [[MPPInteractiveSegmenterOptions alloc] init];

  options.baseOptions.modelAssetPath = modelPath;

  return [self initWithOptions:options error:error];
}

- (nullable MPPInteractiveSegmenterResult *)segmentImage:(MPPImage *)image
                                        regionOfInterest:(MPPRegionOfInterest *)regionOfInterest
                                                   error:(NSError **)error {
  return [self segmentImage:image
                    regionOfInterest:regionOfInterest
      shouldCopyOutputMaskPacketData:YES
                               error:error];
}

- (void)segmentImage:(MPPImage *)image
         regionOfInterest:(MPPRegionOfInterest *)regionOfInterest
    withCompletionHandler:(void (^)(MPPInteractiveSegmenterResult *_Nullable result,
                                    NSError *_Nullable error))completionHandler {
  NSError *error = nil;
  MPPInteractiveSegmenterResult *result = [self segmentImage:image
                                            regionOfInterest:regionOfInterest
                              shouldCopyOutputMaskPacketData:NO
                                                       error:&error];
  completionHandler(result, error);
}

#pragma mark - Private

+ (NSArray<NSString *> *)populateLabelsWithGraphConfig:(const CalculatorGraphConfig &)graphConfig
                                                 error:(NSError **)error {
  bool found_tensor_to_segmentation_calculator = false;

  NSMutableArray<NSString *> *labels =
      [NSMutableArray arrayWithCapacity:(NSUInteger)graphConfig.node_size()];
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

- (nullable MPPInteractiveSegmenterResult *)segmentImage:(MPPImage *)image
                                        regionOfInterest:(MPPRegionOfInterest *)regionOfInterest
                          shouldCopyOutputMaskPacketData:(BOOL)shouldCopyMaskPacketData
                                                   error:(NSError **)error {
  std::optional<PacketMap> inputPacketMap = [_visionTaskRunner inputPacketMapWithMPPImage:image
                                                                         regionOfInterest:CGRectZero
                                                                                    error:error];

  if (!inputPacketMap.has_value()) {
    return nil;
  }
  std::optional<Packet> renderDataPacket =
      [MPPVisionPacketCreator createRenderDataPacketWithRegionOfInterest:regionOfInterest
                                                                   error:error];

  if (!renderDataPacket.has_value()) {
    return nil;
  }

  inputPacketMap->insert({kRoiInStreamName.cppString, renderDataPacket.value()});

  std::optional<PacketMap> outputPacketMap =
      [_visionTaskRunner processPacketMap:inputPacketMap.value() error:error];

  return [MPPInteractiveSegmenter
      interactiveSegmenterResultWithOptionalOutputPacketMap:outputPacketMap
                                   shouldCopyMaskPacketData:shouldCopyMaskPacketData];
}

+ (nullable MPPInteractiveSegmenterResult *)
    interactiveSegmenterResultWithOptionalOutputPacketMap:
        (std::optional<PacketMap> &)outputPacketMap
                                 shouldCopyMaskPacketData:(BOOL)shouldCopyMaskPacketData {
  if (!outputPacketMap.has_value()) {
    return nil;
  }

  PacketMap &outputPacketMapValue = outputPacketMap.value();

  return [MPPInteractiveSegmenterResult
      interactiveSegmenterResultWithConfidenceMasksPacket:outputPacketMapValue
                                                              [kConfidenceMasksStreamName.cppString]
                                       categoryMaskPacket:outputPacketMapValue
                                                              [kCategoryMaskStreamName.cppString]
                                      qualityScoresPacket:outputPacketMapValue
                                                              [kQualityScoresStreamName.cppString]
                                  timestampInMilliseconds:outputPacketMapValue[kImageOutStreamName
                                                                                   .cppString]
                                                              .Timestamp()
                                                              .Value() /
                                                          kMicrosecondsPerMillisecond
                                 shouldCopyMaskPacketData:shouldCopyMaskPacketData];
}

@end
