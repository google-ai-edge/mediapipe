// Copyright 2019 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_PORT_CANONICAL_ERRORS_H_
#define MEDIAPIPE_PORT_CANONICAL_ERRORS_H_

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace mediapipe {

ABSL_DEPRECATED("Use absl::AlreadyExistsError directly")
inline absl::Status AlreadyExistsError(absl::string_view message) {
  return absl::AlreadyExistsError(message);
}

ABSL_DEPRECATED("Use absl::CancelledError directly")
inline absl::Status CancelledError() { return absl::CancelledError(); }

ABSL_DEPRECATED("Use absl::CancelledError directly")
inline absl::Status CancelledError(absl::string_view message) {
  return absl::CancelledError(message);
}

ABSL_DEPRECATED("Use absl::InternalError directly")
inline absl::Status InternalError(absl::string_view message) {
  return absl::InternalError(message);
}

ABSL_DEPRECATED("Use absl::InvalidArgumentError directly")
inline absl::Status InvalidArgumentError(absl::string_view message) {
  return absl::InvalidArgumentError(message);
}

ABSL_DEPRECATED("Use absl::FailedPreconditionError directly")
inline absl::Status FailedPreconditionError(absl::string_view message) {
  return absl::FailedPreconditionError(message);
}

ABSL_DEPRECATED("Use absl::NotFoundError directly")
inline absl::Status NotFoundError(absl::string_view message) {
  return absl::NotFoundError(message);
}

ABSL_DEPRECATED("Use absl::OutOfRangeError directly")
inline absl::Status OutOfRangeError(absl::string_view message) {
  return absl::OutOfRangeError(message);
}

ABSL_DEPRECATED("Use absl::PermissionDeniedError directly")
inline absl::Status PermissionDeniedError(absl::string_view message) {
  return absl::PermissionDeniedError(message);
}
ABSL_DEPRECATED("Use absl::UnavailableError directly")
inline absl::Status UnavailableError(absl::string_view message) {
  return absl::UnavailableError(message);
}

ABSL_DEPRECATED("Use absl::UnimplementedError directly")
inline absl::Status UnimplementedError(absl::string_view message) {
  return absl::UnimplementedError(message);
}

ABSL_DEPRECATED("Use absl::UnknownError directly")
inline absl::Status UnknownError(absl::string_view message) {
  return absl::UnknownError(message);
}

ABSL_DEPRECATED("Use absl::IsCancelled directly")
inline bool IsCancelled(const absl::Status& status) {
  return absl::IsCancelled(status);
}

ABSL_DEPRECATED("Use absl::IsNotFound directly")
inline bool IsNotFound(const absl::Status& status) {
  return absl::IsNotFound(status);
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_PORT_CANONICAL_ERRORS_H_
