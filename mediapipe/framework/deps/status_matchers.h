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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mediapipe/framework/deps/status.h"

namespace mediapipe {

// Monomorphic implementation of matcher IsOk() for a given type T.
// T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoIsOkMatcherImpl : public testing::MatcherInterface<T> {
 public:
  void DescribeTo(std::ostream* os) const override { *os << "is OK"; }
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "is not OK";
  }
  bool MatchAndExplain(T actual_value,
                       testing::MatchResultListener*) const override {
    return actual_value.ok();
  }
};

// Implements IsOk() as a polymorphic matcher.
class IsOkMatcher {
 public:
  template <typename T>
  operator testing::Matcher<T>() const {  // NOLINT
    return testing::Matcher<T>(new MonoIsOkMatcherImpl<T>());
  }
};

// Returns a gMock matcher that matches a Status or StatusOr<> which is OK.
inline IsOkMatcher IsOk() { return IsOkMatcher(); }

}  // namespace mediapipe

// Macros for testing the results of functions that return mediapipe::Status or
// mediapipe::StatusOr<T> (for any type T).
#define MP_EXPECT_OK(expression) EXPECT_THAT(expression, mediapipe::IsOk())
#define MP_ASSERT_OK(expression) ASSERT_THAT(expression, mediapipe::IsOk())

#endif  // MEDIAPIPE_DEPS_STATUS_MATCHERS_H_
