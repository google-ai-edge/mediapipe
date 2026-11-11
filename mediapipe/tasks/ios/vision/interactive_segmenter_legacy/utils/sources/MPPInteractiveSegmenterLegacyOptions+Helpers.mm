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

#import "mediapipe/tasks/ios/vision/interactive_segmenter_legacy/utils/sources/MPPInteractiveSegmenterLegacyOptions+Helpers.h"

#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include <cstdlib>
#include <cstring>

#include "mediapipe/tasks/c/vision/interactive_segmenter_legacy/interactive_segmenter_legacy.h"

@implementation MPPInteractiveSegmenterLegacyOptions (Helpers)

- (void)copyToCOptions:(struct MpInteractiveSegmenterLegacyOptions *)cOptions {
  if (!cOptions) {
    return;
  }
  *cOptions = {};
  [self.baseOptions copyToCOptions:&cOptions->base_options];
  const char *locale = self.displayNamesLocale.UTF8String;
  cOptions->display_names_locale = locale ? strdup(locale) : nullptr;
  cOptions->output_confidence_masks = self.outputConfidenceMasks ? true : false;
  cOptions->output_category_mask = self.outputCategoryMask ? true : false;
}

+ (void)freeCOptions:(struct MpInteractiveSegmenterLegacyOptions *)cOptions {
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
