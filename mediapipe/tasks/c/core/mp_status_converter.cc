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

#include "mediapipe/tasks/c/core/mp_status_converter.h"

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/tasks/c/core/mp_status.h"

namespace mediapipe::tasks::c::core {

MpStatus ToMpStatus(absl::Status status) {
  switch (status.code()) {
    case absl::StatusCode::kOk:
      return kMpOk;
    case absl::StatusCode::kCancelled:
      return kMpCancelled;
    case absl::StatusCode::kUnknown:
      return kMpUnknown;
    case absl::StatusCode::kInvalidArgument:
      return kMpInvalidArgument;
    case absl::StatusCode::kDeadlineExceeded:
      return kMpDeadlineExceeded;
    case absl::StatusCode::kNotFound:
      return kMpNotFound;
    case absl::StatusCode::kAlreadyExists:
      return kMpAlreadyExists;
    case absl::StatusCode::kPermissionDenied:
      return kMpPermissionDenied;
    case absl::StatusCode::kResourceExhausted:
      return kMpResourceExhausted;
    case absl::StatusCode::kFailedPrecondition:
      return kMpFailedPrecondition;
    case absl::StatusCode::kAborted:
      return kMpAborted;
    case absl::StatusCode::kOutOfRange:
      return kMpOutOfRange;
    case absl::StatusCode::kUnimplemented:
      return kMpUnimplemented;
    case absl::StatusCode::kInternal:
      return kMpInternal;
    case absl::StatusCode::kUnavailable:
      return kMpUnavailable;
    case absl::StatusCode::kDataLoss:
      return kMpDataLoss;
    case absl::StatusCode::kUnauthenticated:
      return kMpUnauthenticated;
    default:
      return kMpUnknown;
  }
}

}  // namespace mediapipe::tasks::c::core
