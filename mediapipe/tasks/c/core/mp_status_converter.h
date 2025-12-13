// Copyright 2025 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_TASKS_C_CORE_MP_STATUS_COVERNTER_H_
#define MEDIAPIPE_TASKS_C_CORE_MP_STATUS_COVERNTER_H_

#include "absl/status/status.h"
#include "mediapipe/tasks/c/core/mp_status.h"

namespace mediapipe::tasks::c::core {

MpStatus ToMpStatus(absl::Status status);

// Handles an absl::Status and returns an MpStatus.
// If error_msg is not null, the error message is copied to the error_msg
// buffer. Otherwise, the error message is logged.
MpStatus HandleStatus(absl::Status status, char** error_msg);

}  // namespace mediapipe::tasks::c::core

#endif  // MEDIAPIPE_TASKS_C_CORE_MP_STATUS_COVERNTER_H_
