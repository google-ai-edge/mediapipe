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

#import "mediapipe/tasks/ios/vision/image_segmenter/utils/sources/MPPImageSegmenterOptions+Helpers.h"

#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include <cstdlib>
#include <cstring>

#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter.h"

@implementation MPPImageSegmenterOptions (Helpers)

- (void)copyToCOptions:(struct MpImageSegmenterOptions *)cOptions {
  if (!cOptions) {
    return;
  }
  *cOptions = {};
  [self.baseOptions copyToCOptions:&cOptions->base_options];
  switch (self.runningMode) {
    case MPPRunningModeImage:
      cOptions->running_mode = MP_RUNNING_MODE_IMAGE;
      break;
    case MPPRunningModeVideo:
      cOptions->running_mode = MP_RUNNING_MODE_VIDEO;
      break;
    case MPPRunningModeLiveStream:
      cOptions->running_mode = MP_RUNNING_MODE_LIVE_STREAM;
      break;
  }
  const char *locale = self.displayNamesLocale.UTF8String;
  cOptions->display_names_locale = strdup(locale ? locale : "");
  cOptions->output_confidence_masks = self.outputConfidenceMasks ? true : false;
  cOptions->output_category_mask = self.outputCategoryMask ? true : false;
  cOptions->result_callback = nullptr;
}

+ (void)freeCOptions:(struct MpImageSegmenterOptions *)cOptions {
  if (!cOptions) {
    return;
  }
  [MPPBaseOptions freeCOptions:&cOptions->base_options];
  if (cOptions->display_names_locale) {
    free(const_cast<char *>(cOptions->display_names_locale));
    cOptions->display_names_locale = nullptr;
  }
}

@end
