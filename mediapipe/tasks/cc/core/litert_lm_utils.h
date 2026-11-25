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

#ifndef MEDIAPIPE_TASKS_CC_CORE_LITERT_LM_UTILS_H_
#define MEDIAPIPE_TASKS_CC_CORE_LITERT_LM_UTILS_H_

#include "absl/status/status.h"        // from @com_google_absl
#include "absl/status/statusor.h"      // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "mediapipe/tasks/cc/core/base_options.h"
#include "runtime/executor/executor_settings_base.h"  // from @litert_lm

namespace mediapipe::tasks::core {

// Maps a MediaPipe Tasks `BaseOptions::Delegate` to the corresponding
// LiteRT-LM `Backend`.
litert::lm::Backend ToLiteRtLmBackend(BaseOptions::Delegate delegate);

// Validates that `base_options` specifies a supported LiteRT-LM model asset
// (either `model_asset_path` or a whole-file `model_asset_descriptor_meta.fd`,
// but not both; in-memory `model_asset_buffer` and sliced FD `offset`/`length`
// are rejected). `option_name` is included in error messages to identify the
// field.
absl::Status ValidateLiteRtLmModelAsset(
    const BaseOptions& base_options,
    absl::string_view option_name = "base_options");

// Creates `litert::lm::ModelAssets` from `base_options.model_asset_path` or a
// duplicated whole-file `base_options.model_asset_descriptor_meta.fd` so the
// caller retains ownership of its FD.
absl::StatusOr<litert::lm::ModelAssets> CreateLiteRtLmModelAssets(
    const BaseOptions& base_options,
    absl::string_view option_name = "base_options");

}  // namespace mediapipe::tasks::core

#endif  // MEDIAPIPE_TASKS_CC_CORE_LITERT_LM_UTILS_H_
