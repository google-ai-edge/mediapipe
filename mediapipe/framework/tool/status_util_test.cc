// Copyright 2018 The MediaPipe Authors.
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

#include <memory>
#include <string>
#include <vector>

#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using testing::ContainerEq;
using testing::Eq;
using testing::HasSubstr;
using testing::IsEmpty;
using testing::Matches;
using testing::Pointwise;

TEST(StatusTest, StatusStopIsNotOk) { EXPECT_FALSE(tool::StatusStop().ok()); }

TEST(StatusTest, Prefix) {
  const std::string base_error_message("error_with_this_string");
  const std::string prefix_error_message("error_with_prefix: ");
  absl::Status base_status =
      absl::Status(absl::StatusCode::kInvalidArgument, base_error_message);
  absl::Status status =
      tool::AddStatusPrefix(prefix_error_message, base_status);
  EXPECT_THAT(status.ToString(), HasSubstr(base_error_message));
  EXPECT_THAT(status.ToString(), HasSubstr(prefix_error_message));
  EXPECT_EQ(absl::StatusCode::kInvalidArgument, status.code());
}

TEST(StatusTest, CombinedStatus) {
  std::vector<absl::Status> errors;
  const std::string prefix_error_message("error_with_prefix: ");
  absl::Status status;

  errors.clear();
  errors.emplace_back(absl::StatusCode::kInvalidArgument,
                      "error_with_this_string");
  errors.emplace_back(absl::StatusCode::kInvalidArgument,
                      "error_with_that_string");
  errors.back().SetPayload("test payload type",
                           absl::Cord(absl::string_view("hello")));
  status = tool::CombinedStatus(prefix_error_message, errors);
  EXPECT_THAT(status.ToString(), HasSubstr(std::string(errors[0].message())));
  EXPECT_THAT(status.ToString(), HasSubstr(std::string(errors[1].message())));
  EXPECT_THAT(status.ToString(), HasSubstr(prefix_error_message));
  EXPECT_EQ(absl::StatusCode::kInvalidArgument, status.code());

  errors.clear();
  errors.emplace_back(absl::StatusCode::kNotFound, "error_with_this_string");
  errors.emplace_back(absl::StatusCode::kInvalidArgument,
                      "error_with_that_string");
  status = tool::CombinedStatus(prefix_error_message, errors);
  EXPECT_THAT(status.ToString(), HasSubstr(std::string(errors[0].message())));
  EXPECT_THAT(status.ToString(), HasSubstr(std::string(errors[1].message())));
  EXPECT_THAT(status.ToString(), HasSubstr(prefix_error_message));
  EXPECT_EQ(absl::StatusCode::kUnknown, status.code());
  errors.clear();
  errors.emplace_back(absl::StatusCode::kOk, "error_with_this_string");
  errors.emplace_back(absl::StatusCode::kInvalidArgument,
                      "error_with_that_string");
  status = tool::CombinedStatus(prefix_error_message, errors);
  EXPECT_THAT(status.ToString(), HasSubstr(std::string(errors[1].message())));
  EXPECT_THAT(status.ToString(), HasSubstr(prefix_error_message));
  EXPECT_EQ(absl::StatusCode::kInvalidArgument, status.code());

  errors.clear();
  errors.emplace_back(absl::StatusCode::kOk, "error_with_this_string");
  errors.emplace_back(absl::StatusCode::kOk, "error_with_that_string");
  MP_EXPECT_OK(tool::CombinedStatus(prefix_error_message, errors));

  errors.clear();
  MP_EXPECT_OK(tool::CombinedStatus(prefix_error_message, errors));
}

// Verify tool::StatusInvalid() and tool::StatusFail() and the alternatives
// recommended by their ABSL_DEPRECATED messages return the same
// absl::Status objects.
TEST(StatusTest, Deprecated) {
  const std::string error_message = "an error message";
  EXPECT_EQ(tool::StatusInvalid(error_message),  // NOLINT
            absl::InvalidArgumentError(error_message));
  EXPECT_EQ(tool::StatusFail(error_message),  // NOLINT
            absl::UnknownError(error_message));
}

}  // namespace
}  // namespace mediapipe
