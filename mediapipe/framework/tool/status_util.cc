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

#include "mediapipe/framework/tool/status_util.h"

#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace mediapipe {
namespace tool {

::mediapipe::Status StatusInvalid(const std::string& message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kInvalidArgument,
                             message);
}

::mediapipe::Status StatusFail(const std::string& message) {
  return ::mediapipe::Status(::mediapipe::StatusCode::kUnknown, message);
}

::mediapipe::Status StatusStop() {
  return ::mediapipe::Status(::mediapipe::StatusCode::kOutOfRange,
                             "::mediapipe::tool::StatusStop()");
}

::mediapipe::Status AddStatusPrefix(const std::string& prefix,
                                    const ::mediapipe::Status& status) {
  return ::mediapipe::Status(status.code(),
                             absl::StrCat(prefix, status.message()));
}

::mediapipe::Status CombinedStatus(
    const std::string& general_comment,
    const std::vector<::mediapipe::Status>& statuses) {
  // The final error code is ::mediapipe::StatusCode::kUnknown if not all
  // the error codes are the same.  Otherwise it is the same error code
  // as all of the (non-OK) statuses.  If statuses is empty or they are
  // all OK, then ::mediapipe::OkStatus() is returned.
  ::mediapipe::StatusCode error_code = ::mediapipe::StatusCode::kOk;
  std::vector<std::string> errors;
  for (const ::mediapipe::Status& status : statuses) {
    if (!status.ok()) {
      errors.emplace_back(status.message());
      if (error_code == ::mediapipe::StatusCode::kOk) {
        error_code = status.code();
      } else if (error_code != status.code()) {
        error_code = ::mediapipe::StatusCode::kUnknown;
      }
    }
  }
  return ::mediapipe::Status(
      error_code,
      absl::StrCat(general_comment, "\n", absl::StrJoin(errors, "\n")));
}

}  // namespace tool
}  // namespace mediapipe
