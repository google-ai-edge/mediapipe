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

#ifndef MEDIAPIPE_TASKS_C_CORE_MP_STATUS_H_
#define MEDIAPIPE_TASKS_C_CORE_MP_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

// Status codes for MediaPipe C API functions.
// These codes are aligned with absl::StatusCode.
typedef enum MpStatus {
  kMpOk = 0,
  kMpCancelled = 1,
  kMpUnknown = 2,
  kMpInvalidArgument = 3,
  kMpDeadlineExceeded = 4,
  kMpNotFound = 5,
  kMpAlreadyExists = 6,
  kMpPermissionDenied = 7,
  kMpResourceExhausted = 8,
  kMpFailedPrecondition = 9,
  kMpAborted = 10,
  kMpOutOfRange = 11,
  kMpUnimplemented = 12,
  kMpInternal = 13,
  kMpUnavailable = 14,
  kMpDataLoss = 15,
  kMpUnauthenticated = 16,
} MpStatus;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_TASKS_C_CORE_MP_STATUS_H_
