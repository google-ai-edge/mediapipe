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

#import "mediapipe/tasks/ios/vision/interactive_segmenter_legacy/sources/MPPInteractiveSegmenterLegacy.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPRegionOfInterest+Helpers.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"
#import "mediapipe/tasks/ios/vision/interactive_segmenter_legacy/utils/sources/MPPInteractiveSegmenterLegacyOptions+Helpers.h"
#import "mediapipe/tasks/ios/vision/interactive_segmenter_legacy/utils/sources/MPPInteractiveSegmenterLegacyResult+Helpers.h"

#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/interactive_segmenter_legacy/interactive_segmenter_legacy.h"

@interface MPPInteractiveSegmenterLegacy () {
  MpInteractiveSegmenterLegacyPtr _segmenter;
  MPPVisionTaskRunner *_taskRunner;
  NSArray<NSString *> *_labels;
}

- (BOOL)populateCResult:(MpImageSegmenterResult *)cResult
              withImage:(MPPImage *)image
       regionOfInterest:(MPPRegionOfInterest *)regionOfInterest
                  error:(NSError **)error;
@end

@implementation MPPInteractiveSegmenterLegacy

- (instancetype)initWithOptions:(MPPInteractiveSegmenterLegacyOptions *)options
                          error:(NSError **)error {
  self = [super init];
  if (self) {
    if (!options) {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"options cannot be nil."];
      return nil;
    }
    _taskRunner = [[MPPVisionTaskRunner alloc] initWithRunningMode:MPPRunningModeImage
                                                        roiAllowed:YES
                                                   normRectAllowed:YES
                                                       hasDelegate:NO
                                                             error:error];
    if (!_taskRunner) {
      return nil;
    }

    MpInteractiveSegmenterLegacyOptions cOptions = {};
    [options copyToCOptions:&cOptions];

    char *errorMsg = nullptr;
    MpStatus status = MpInteractiveSegmenterLegacyCreate(&cOptions, &_segmenter, &errorMsg);
    [MPPInteractiveSegmenterLegacyOptions freeCOptions:&cOptions];
    if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
      return nil;
    }

    MpStringList labelList = {};
    errorMsg = nullptr;
    status = MpInteractiveSegmenterLegacyGetLabels(_segmenter, &labelList, &errorMsg);
    if (![MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error]) {
      MpInteractiveSegmenterLegacyClose(_segmenter, nullptr);
      _segmenter = nullptr;
      return nil;
    }
    NSMutableArray<NSString *> *labels =
        [NSMutableArray arrayWithCapacity:labelList.num_strings];
    for (int i = 0; i < labelList.num_strings; ++i) {
      NSString *label = (labelList.strings && labelList.strings[i])
                            ? [NSString stringWithUTF8String:labelList.strings[i]]
                            : nil;
      [labels addObject:label ?: @""];
    }
    _labels = [labels copy];
    MpStringListFree(&labelList);
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  MPPInteractiveSegmenterLegacyOptions *options =
      [[MPPInteractiveSegmenterLegacyOptions alloc] init];
  options.baseOptions.modelAssetPath = modelPath;
  return [self initWithOptions:options error:error];
}

- (nullable MPPInteractiveSegmenterLegacyResult *)segmentImage:(MPPImage *)image
                                              regionOfInterest:
                                                  (MPPRegionOfInterest *)regionOfInterest
                                                         error:(NSError **)error {
  MpImageSegmenterResult cResult = {};
  if (![self populateCResult:&cResult
                   withImage:image
            regionOfInterest:regionOfInterest
                       error:error]) {
    return nil;
  }
  MPPInteractiveSegmenterLegacyResult *result = [MPPInteractiveSegmenterLegacyResult
      interactiveSegmenterLegacyResultWithCResult:cResult
                          timestampInMilliseconds:0
                         shouldCopyMaskPacketData:YES];
  MpInteractiveSegmenterLegacyCloseResult(&cResult);
  return result;
}

- (void)segmentImage:(MPPImage *)image
         regionOfInterest:(MPPRegionOfInterest *)regionOfInterest
    withCompletionHandler:(void (^)(MPPInteractiveSegmenterLegacyResult *_Nullable result,
                                    NSError *_Nullable error))completionHandler {
  if (!completionHandler) {
    return;
  }
  NSError *error = nil;
  MpImageSegmenterResult cResult = {};
  if (![self populateCResult:&cResult
                   withImage:image
            regionOfInterest:regionOfInterest
                       error:&error]) {
    completionHandler(nil, error);
    return;
  }
  MPPInteractiveSegmenterLegacyResult *result = [MPPInteractiveSegmenterLegacyResult
      interactiveSegmenterLegacyResultWithCResult:cResult
                          timestampInMilliseconds:0
                         shouldCopyMaskPacketData:NO];
  completionHandler(result, nil);
  MpInteractiveSegmenterLegacyCloseResult(&cResult);
}

- (void)dealloc {
  if (_segmenter) {
    MpInteractiveSegmenterLegacyClose(_segmenter, nullptr);
    _segmenter = nullptr;
  }
}

#pragma mark - Private

- (BOOL)populateCResult:(MpImageSegmenterResult *)cResult
              withImage:(MPPImage *)image
       regionOfInterest:(MPPRegionOfInterest *)regionOfInterest
                  error:(NSError **)error {
  if (!regionOfInterest) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"`regionOfInterest` cannot be `nil`."];
    return NO;
  }

  struct MpImageProcessingOptions imageProcessingOptions = {};
  ScopedMpImage mpImage = [_taskRunner createMpImageWithMPPImage:image
                                                regionOfInterest:CGRectZero
                                          imageProcessingOptions:&imageProcessingOptions
                                                           error:error];
  if (!mpImage) {
    return NO;
  }

  MpRegionOfInterest cRoi = {};
  if (![regionOfInterest copyToCRegionOfInterest:&cRoi error:error]) {
    return NO;
  }

  char *errorMsg = nullptr;
  MpStatus status = MpInteractiveSegmenterLegacySegmentImage(
      _segmenter, mpImage, &cRoi, &imageProcessingOptions, cResult, &errorMsg);
  [MPPRegionOfInterest freeCRegionOfInterest:&cRoi];

  return [MPPCommonUtils checkCError:status errorMessage:errorMsg toError:error];
}

@end
