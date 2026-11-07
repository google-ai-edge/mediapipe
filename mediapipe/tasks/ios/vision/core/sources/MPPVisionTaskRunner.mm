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

#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/vision/core/utils/sources/MPPImage+Utils.h"

#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"

/** Rotation degrees for a 90 degree rotation to the right. */
static const NSInteger kMPPOrientationDegreesRight = -270;

/** Rotation degrees for a 180 degree rotation. */
static const NSInteger kMPPOrientationDegreesDown = -180;

/** Rotation degrees for a 90 degree rotation to the left. */
static const NSInteger kMPPOrientationDegreesLeft = -90;

static NSString *const kTaskPrefix = @"com.mediapipe.tasks.vision";

@interface MPPVisionTaskRunner () {
  MPPRunningMode _runningMode;
  BOOL _roiAllowed;
  BOOL _normRectAllowed;
}
@end

@implementation MPPVisionTaskRunner

- (nullable instancetype)initWithRunningMode:(MPPRunningMode)runningMode
                                  roiAllowed:(BOOL)roiAllowed
                             normRectAllowed:(BOOL)normRectAllowed
                                 hasDelegate:(BOOL)hasDelegate
                                       error:(NSError **)error {
  switch (runningMode) {
    case MPPRunningModeImage:
    case MPPRunningModeVideo: {
      if (hasDelegate) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInvalidArgumentError
                              description:@"The vision task is in image or video mode. The "
                                          @"delegate must not be set in the task's options."];
        return nil;
      }
      break;
    }
    case MPPRunningModeLiveStream: {
      if (!hasDelegate) {
        [MPPCommonUtils
            createCustomError:error
                     withCode:MPPTasksErrorCodeInvalidArgumentError
                  description:
                      @"The vision task is in live stream mode. An object must be set as the "
                      @"delegate of the task in its options to ensure asynchronous delivery of "
                      @"results."];
        return nil;
      }
      break;
    }
    default: {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Unrecognized running mode"];
      return nil;
    }
  }

  self = [super init];
  if (self) {
    _runningMode = runningMode;
    _roiAllowed = roiAllowed;
    _normRectAllowed = normRectAllowed;
  }
  return self;
}

- (BOOL)validateRunningMode:(MPPRunningMode)expectedMode error:(NSError **)error {
  if (_runningMode != expectedMode) {
    NSString *expectedModeName = @"";
    switch (expectedMode) {
      case MPPRunningModeImage:
        expectedModeName = @"image";
        break;
      case MPPRunningModeVideo:
        expectedModeName = @"video";
        break;
      case MPPRunningModeLiveStream:
        expectedModeName = @"live stream";
        break;
    }
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:[NSString stringWithFormat:@"The vision task is not initialized with "
                                                     @"%@ mode. Current Running Mode: %@",
                                                     expectedModeName,
                                                     MPPRunningModeDisplayName(_runningMode)]];
    return NO;
  }
  return YES;
}

- (nullable MpImagePtr)createMpImageWithMPPImage:(MPPImage *)image
                                regionOfInterest:(CGRect)roi
                          imageProcessingOptions:(struct MpImageProcessingOptions *)outOptions
                                           error:(NSError **)error {
  if (!image) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"`image` cannot be `nil`."];
    return nullptr;
  }

  if (!_normRectAllowed && image.orientation != UIImageOrientationUp) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Unsupported UIImageOrientation. This task only supports "
                                      @"images with `imageOrientation = UIImageOrientationUp`"];
    return nullptr;
  }

  if ((!_normRectAllowed || !_roiAllowed) && !CGRectEqualToRect(roi, CGRectZero)) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInternalError
                          description:@"This task doesn't support region-of-interest."];
    return nullptr;
  }

  int rotationDegrees = 0;
  switch (image.orientation) {
    case UIImageOrientationUp:
      rotationDegrees = 0;
      break;
    case UIImageOrientationRight:
      rotationDegrees = kMPPOrientationDegreesRight;
      break;
    case UIImageOrientationDown:
      rotationDegrees = kMPPOrientationDegreesDown;
      break;
    case UIImageOrientationLeft:
      rotationDegrees = kMPPOrientationDegreesLeft;
      break;
    default:
      [MPPCommonUtils
          createCustomError:error
                   withCode:MPPTasksErrorCodeInvalidArgumentError
                description:
                    @"Unsupported UIImageOrientation. `imageOrientation` cannot be equal to "
                    @"any of the mirrored orientations "
                    @"(`UIImageOrientationUpMirrored`,`UIImageOrientationDownMirrored`,`"
                    @"UIImageOrientationLeftMirrored`,`UIImageOrientationRightMirrored`)"];
      return nullptr;
  }

  if (outOptions) {
    outOptions->rotation_degrees = rotationDegrees;
    if (!CGRectEqualToRect(roi, CGRectZero)) {
      outOptions->has_region_of_interest = 1;
      outOptions->region_of_interest = {
          .left = static_cast<float>(roi.origin.x),
          .top = static_cast<float>(roi.origin.y),
          .bottom = static_cast<float>(roi.origin.y + roi.size.height),
          .right = static_cast<float>(roi.origin.x + roi.size.width),
      };
    } else {
      outOptions->has_region_of_interest = 0;
      outOptions->region_of_interest = {};
    }
  }

  return [image mpImageWithError:error];
}

+ (const char *)uniqueDispatchQueueNameWithSuffix:(NSString *)suffix {
  return [NSString stringWithFormat:@"%@.%@_%@", kTaskPrefix, suffix, [NSString uuidString]]
      .UTF8String;
}

@end
