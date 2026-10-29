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

#ifndef MEDIAPIPE_DEPS_STATUS_BUILDER_H_
#define MEDIAPIPE_DEPS_STATUS_BUILDER_H_

#include "absl/status/status.h"
#include "absl/status/status_builder.h"
#include "absl/types/source_location.h"
#include "mediapipe/framework/deps/source_location.h"

namespace mediapipe {

using StatusBuilder = ::absl::StatusBuilder;

inline StatusBuilder AlreadyExistsErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kAlreadyExists, location);
}

inline StatusBuilder FailedPreconditionErrorBuilder(
    absl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kFailedPrecondition, location);
}

inline StatusBuilder InternalErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kInternal, location);
}

inline StatusBuilder InvalidArgumentErrorBuilder(
    absl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kInvalidArgument, location);
}

inline StatusBuilder NotFoundErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kNotFound, location);
}

inline StatusBuilder UnavailableErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kUnavailable, location);
}

inline StatusBuilder UnimplementedErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kUnimplemented, location);
}

inline StatusBuilder UnknownErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(absl::StatusCode::kUnknown, location);
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_DEPS_STATUS_BUILDER_H_
