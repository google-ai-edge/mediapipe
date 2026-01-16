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

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/no_destructor.h"

namespace mediapipe {
namespace tool {

absl::Status StatusInvalid(absl::string_view message) {
  return absl::Status(absl::StatusCode::kInvalidArgument, message);
}

absl::Status StatusFail(absl::string_view message) {
  return absl::Status(absl::StatusCode::kUnknown, message);
}

const absl::Status& StatusStop() {
  static const NoDestructor<absl::Status> kStatusStop(
      absl::StatusCode::kOutOfRange, "mediapipe::tool::StatusStop()");
  return *kStatusStop;
}

absl::Status AddStatusPrefix(absl::string_view prefix,
                             const absl::Status& status) {
  return absl::Status(status.code(), absl::StrCat(prefix, status.message()));
}

absl::Status CombinedStatus(absl::string_view general_comment,
                            const std::vector<absl::Status>& statuses) {
  // The final error code is absl::StatusCode::kUnknown if not all
  // the error codes are the same.  Otherwise it is the same error code
  // as all of the (non-OK) statuses.  If statuses is empty or they are
  // all OK, then absl::OkStatus() is returned.
  absl::StatusCode error_code = absl::StatusCode::kOk;
  std::vector<std::string> errors;
  for (const absl::Status& status : statuses) {
    if (!status.ok()) {
      errors.emplace_back(status.message());
      if (error_code == absl::StatusCode::kOk) {
        error_code = status.code();
      } else if (error_code != status.code()) {
        error_code = absl::StatusCode::kUnknown;
      }
    }
  }
  if (error_code == absl::StatusCode::kOk) return absl::OkStatus();
  absl::Status combined;
  combined = absl::Status(
      error_code,
      absl::StrCat(general_comment, "\n", absl::StrJoin(errors, "\n")));
  return combined;
}

}  // namespace tool
}  // namespace mediapipe
