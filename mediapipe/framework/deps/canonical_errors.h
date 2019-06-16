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

#ifndef MEDIAPIPE_DEPS_CANONICAL_ERRORS_H_
#define MEDIAPIPE_DEPS_CANONICAL_ERRORS_H_

#include "mediapipe/framework/deps/status.h"

namespace mediapipe {

// Each of the functions below creates a canonical error with the given
// message. The error code of the returned status object matches the name of
// the function.
inline ::mediapipe::Status AlreadyExistsError(absl::string_view message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kAlreadyExists, message);
}

inline ::mediapipe::Status CancelledError() {
  return ::mediapipe::Status(::mediapipe::StatusCode::kCancelled, "");
}

inline ::mediapipe::Status CancelledError(absl::string_view message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kCancelled, message);
}

inline ::mediapipe::Status InternalError(absl::string_view message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kInternal, message);
}

inline ::mediapipe::Status InvalidArgumentError(absl::string_view message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kInvalidArgument,
                             message);
}

inline ::mediapipe::Status FailedPreconditionError(absl::string_view message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kFailedPrecondition,
                             message);
}

inline ::mediapipe::Status NotFoundError(absl::string_view message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kNotFound, message);
}

inline ::mediapipe::Status OutOfRangeError(absl::string_view message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kOutOfRange, message);
}

inline ::mediapipe::Status PermissionDeniedError(absl::string_view message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kPermissionDenied,
                             message);
}

inline ::mediapipe::Status UnimplementedError(absl::string_view message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kUnimplemented, message);
}

inline ::mediapipe::Status UnknownError(absl::string_view message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kUnknown, message);
}

inline ::mediapipe::Status UnavailableError(absl::string_view message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kUnavailable, message);
}

inline bool IsCancelled(const ::mediapipe::Status& status) {
  return status.code() == ::mediapipe::StatusCode::kCancelled;
}

inline bool IsNotFound(const ::mediapipe::Status& status) {
  return status.code() == ::mediapipe::StatusCode::kNotFound;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_DEPS_CANONICAL_ERRORS_H_
