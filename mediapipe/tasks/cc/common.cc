/* Copyright 2022 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/common.h"

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace mediapipe {
namespace tasks {

absl::Status CreateStatusWithPayload(
    absl::StatusCode canonical_code, absl::string_view message,
    MediaPipeTasksStatus mediapipe_tasks_code) {
  // NOTE: Ignores `message` if the canonical code is ok.
  absl::Status status = absl::Status(canonical_code, message);
  // NOTE: Does nothing if the canonical code is ok.
  status.SetPayload(kMediaPipeTasksPayload,
                    absl::Cord(absl::StrCat(mediapipe_tasks_code)));
  return status;
}

absl::Status AddPayload(absl::Status status, absl::string_view message,
                        MediaPipeTasksStatus mediapipe_tasks_code) {
  if (status.ok()) return status;
  // Attaches a new payload with the MediaPipeTasksStatus key to the status.
  status.SetPayload(kMediaPipeTasksPayload,
                    absl::Cord(absl::StrCat(mediapipe_tasks_code)));
  return status;
}

}  // namespace tasks
}  // namespace mediapipe
