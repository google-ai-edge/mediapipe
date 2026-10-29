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

#ifndef MEDIAPIPE_DEPS_STATUS_MATCHERS_H_
#define MEDIAPIPE_DEPS_STATUS_MATCHERS_H_

#include "absl/status/status.h"
#include "absl/status/status_builder.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mediapipe {

// Deprecated aliases kept for compatibility. Prefer using
// `::absl_testing::IsOk`, `::absl_testing::IsOkAndHolds` and
// `::absl_testing::StatusIs` directly.
using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

}  // namespace mediapipe

// Macros for testing the results of functions that return absl::Status or
// absl::StatusOr<T> (for any type T).
#define MP_EXPECT_OK(expression) EXPECT_THAT(expression, ::absl_testing::IsOk())
#define MP_ASSERT_OK(expression) ASSERT_THAT(expression, ::absl_testing::IsOk())

#define STATUS_MACROS_IMPL_CONCAT_INNER_(x, y) x##y
#define STATUS_MACROS_IMPL_CONCAT_(x, y) STATUS_MACROS_IMPL_CONCAT_INNER_(x, y)

#undef MP_ASSERT_OK_AND_ASSIGN
#define MP_ASSERT_OK_AND_ASSIGN(lhs, rexpr) \
  MP_ASSERT_OK_AND_ASSIGN_IMPL_(            \
      STATUS_MACROS_IMPL_CONCAT_(_status_or_value, __LINE__), lhs, rexpr)

inline void MpInternalAssertAddFatalFailure(
    absl::string_view expression, const absl::StatusBuilder& builder) {
  FAIL() << absl::StrCat(expression, " returned error: ",
                         absl::Status(builder).ToString(
                             absl::StatusToStringMode::kWithEverything));
}

#define MP_ASSERT_OK_AND_ASSIGN_IMPL_(statusor, lhs, rexpr) \
  ABSL_ASSIGN_OR_RETURN(lhs, rexpr, MpInternalAssertAddFatalFailure(#rexpr, _));

#endif  // MEDIAPIPE_DEPS_STATUS_MATCHERS_H_
