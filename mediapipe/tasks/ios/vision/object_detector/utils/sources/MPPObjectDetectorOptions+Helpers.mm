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

#import "mediapipe/tasks/ios/vision/object_detector/utils/sources/MPPObjectDetectorOptions+Helpers.h"

#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include <cstdlib>
#include <cstring>

#include "mediapipe/tasks/c/vision/object_detector/object_detector.h"

@implementation MPPObjectDetectorOptions (Helpers)

- (void)copyToCOptions:(struct MpObjectDetectorOptions *)cOptions {
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
  cOptions->display_names_locale =
      self.displayNamesLocale ? strdup(self.displayNamesLocale.UTF8String) : nullptr;
  cOptions->max_results = static_cast<int>(self.maxResults);
  cOptions->score_threshold = self.scoreThreshold;
  cOptions->category_allowlist = nullptr;
  cOptions->category_allowlist_count = 0;
  cOptions->category_denylist = nullptr;
  cOptions->category_denylist_count = 0;
  cOptions->result_callback = nullptr;

  if (self.categoryAllowlist.count > 0) {
    cOptions->category_allowlist_count = static_cast<uint32_t>(self.categoryAllowlist.count);
    cOptions->category_allowlist = new const char *[cOptions->category_allowlist_count];
    for (uint32_t i = 0; i < cOptions->category_allowlist_count; ++i) {
      cOptions->category_allowlist[i] = strdup(self.categoryAllowlist[i].UTF8String);
    }
  }

  if (self.categoryDenylist.count > 0) {
    cOptions->category_denylist_count = static_cast<uint32_t>(self.categoryDenylist.count);
    cOptions->category_denylist = new const char *[cOptions->category_denylist_count];
    for (uint32_t i = 0; i < cOptions->category_denylist_count; ++i) {
      cOptions->category_denylist[i] = strdup(self.categoryDenylist[i].UTF8String);
    }
  }
}

+ (void)freeCOptions:(struct MpObjectDetectorOptions *)cOptions {
  if (!cOptions) {
    return;
  }
  [MPPBaseOptions freeCOptions:&cOptions->base_options];
  if (cOptions->display_names_locale) {
    free(const_cast<char *>(cOptions->display_names_locale));
    cOptions->display_names_locale = nullptr;
  }
  if (cOptions->category_allowlist) {
    for (uint32_t i = 0; i < cOptions->category_allowlist_count; ++i) {
      free(const_cast<char *>(cOptions->category_allowlist[i]));
    }
    delete[] cOptions->category_allowlist;
    cOptions->category_allowlist = nullptr;
  }
  cOptions->category_allowlist_count = 0;
  if (cOptions->category_denylist) {
    for (uint32_t i = 0; i < cOptions->category_denylist_count; ++i) {
      free(const_cast<char *>(cOptions->category_denylist[i]));
    }
    delete[] cOptions->category_denylist;
    cOptions->category_denylist = nullptr;
  }
  cOptions->category_denylist_count = 0;
}

@end
