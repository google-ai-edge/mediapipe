/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/c/components/processors/classifier_options.h"

#include <cstdint>
#include <cstdlib>

extern "C" {

void MpClassifierOptionsFree(struct MpClassifierOptions* options) {
  if (options == nullptr) {
    return;
  }
  free(const_cast<char*>(options->display_names_locale));
  options->display_names_locale = nullptr;

  options->max_results = -1;
  options->score_threshold = 0.0f;

  if (options->category_allowlist != nullptr) {
    for (uint32_t i = 0; i < options->category_allowlist_count; ++i) {
      free(const_cast<char*>(options->category_allowlist[i]));
    }
    free(options->category_allowlist);
    options->category_allowlist = nullptr;
  }
  options->category_allowlist_count = 0;

  if (options->category_denylist != nullptr) {
    for (uint32_t i = 0; i < options->category_denylist_count; ++i) {
      free(const_cast<char*>(options->category_denylist[i]));
    }
    free(options->category_denylist);
    options->category_denylist = nullptr;
  }
  options->category_denylist_count = 0;
}

}  // extern "C"
