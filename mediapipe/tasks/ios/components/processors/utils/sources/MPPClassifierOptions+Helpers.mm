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

#import "mediapipe/tasks/ios/components/processors/utils/sources/MPPClassifierOptions+Helpers.h"

#include <cstdlib>
#include <cstring>

#include "mediapipe/tasks/c/components/processors/classifier_options.h"

@implementation MPPClassifierOptions (Helpers)

- (void)copyToCOptions:(MpClassifierOptions *)cClassifierOptions {
  cClassifierOptions->display_names_locale =
      self.displayNamesLocale ? strdup(self.displayNamesLocale.UTF8String) : nullptr;
  cClassifierOptions->max_results = static_cast<int>(self.maxResults);
  cClassifierOptions->score_threshold = self.scoreThreshold;
  cClassifierOptions->category_allowlist = nullptr;
  cClassifierOptions->category_allowlist_count = 0;
  cClassifierOptions->category_denylist = nullptr;
  cClassifierOptions->category_denylist_count = 0;

  if (self.categoryAllowlist.count > 0) {
    cClassifierOptions->category_allowlist_count =
        static_cast<uint32_t>(self.categoryAllowlist.count);
    cClassifierOptions->category_allowlist = static_cast<const char **>(
        calloc(cClassifierOptions->category_allowlist_count, sizeof(const char *)));
    for (uint32_t i = 0; i < cClassifierOptions->category_allowlist_count; ++i) {
      cClassifierOptions->category_allowlist[i] = strdup(self.categoryAllowlist[i].UTF8String);
    }
  }

  if (self.categoryDenylist.count > 0) {
    cClassifierOptions->category_denylist_count =
        static_cast<uint32_t>(self.categoryDenylist.count);
    cClassifierOptions->category_denylist = static_cast<const char **>(
        calloc(cClassifierOptions->category_denylist_count, sizeof(const char *)));
    for (uint32_t i = 0; i < cClassifierOptions->category_denylist_count; ++i) {
      cClassifierOptions->category_denylist[i] = strdup(self.categoryDenylist[i].UTF8String);
    }
  }
}

+ (void)freeCOptions:(nullable MpClassifierOptions *)cClassifierOptions {
  MpClassifierOptionsFree(cClassifierOptions);
}

@end
