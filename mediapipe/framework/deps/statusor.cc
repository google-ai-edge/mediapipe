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

#include "mediapipe/framework/deps/statusor.h"

#include "absl/base/attributes.h"
#include "mediapipe/framework/deps/canonical_errors.h"
#include "mediapipe/framework/deps/status.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {
namespace internal_statusor {

void Helper::HandleInvalidStatusCtorArg(::mediapipe::Status* status) {
  const char* kMessage =
      "An OK status is not a valid constructor argument to StatusOr<T>";
  LOG(ERROR) << kMessage;
  *status = ::mediapipe::InternalError(kMessage);
}

void Helper::Crash(const ::mediapipe::Status& status) {
  LOG(FATAL) << "Attempting to fetch value instead of handling error "
             << status;
}

}  // namespace internal_statusor
}  // namespace mediapipe
