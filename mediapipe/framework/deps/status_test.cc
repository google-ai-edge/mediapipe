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

#include "mediapipe/framework/deps/status.h"

#include "mediapipe/framework/deps/status_matchers.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {

TEST(Status, OK) {
  EXPECT_EQ(OkStatus().code(), ::mediapipe::StatusCode::kOk);
  EXPECT_EQ(OkStatus().message(), "");
  MP_EXPECT_OK(OkStatus());
  MP_ASSERT_OK(OkStatus());
  EXPECT_EQ(OkStatus(), Status());
  Status s;
  EXPECT_TRUE(s.ok());
}

TEST(DeathStatus, CheckOK) {
  Status status(::mediapipe::StatusCode::kInvalidArgument, "Invalid");
  ASSERT_DEATH(MEDIAPIPE_CHECK_OK(status), "Invalid");
}

TEST(Status, Set) {
  Status status;
  status = Status(::mediapipe::StatusCode::kCancelled, "Error message");
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kCancelled);
  EXPECT_EQ(status.message(), "Error message");
}

TEST(Status, Copy) {
  Status a(::mediapipe::StatusCode::kInvalidArgument, "Invalid");
  Status b(a);
  ASSERT_EQ(a.ToString(), b.ToString());
}

TEST(Status, Assign) {
  Status a(::mediapipe::StatusCode::kInvalidArgument, "Invalid");
  Status b;
  b = a;
  ASSERT_EQ(a.ToString(), b.ToString());
}

TEST(Status, Update) {
  Status s;
  s.Update(OkStatus());
  ASSERT_TRUE(s.ok());
  Status a(::mediapipe::StatusCode::kInvalidArgument, "Invalid");
  s.Update(a);
  ASSERT_EQ(s.ToString(), a.ToString());
  Status b(::mediapipe::StatusCode::kInternal, "Invalid");
  s.Update(b);
  ASSERT_EQ(s.ToString(), a.ToString());
  s.Update(OkStatus());
  ASSERT_EQ(s.ToString(), a.ToString());
  ASSERT_FALSE(s.ok());
}

TEST(Status, EqualsOK) { ASSERT_EQ(OkStatus(), Status()); }

TEST(Status, EqualsSame) {
  Status a(::mediapipe::StatusCode::kInvalidArgument, "Invalid");
  Status b(::mediapipe::StatusCode::kInvalidArgument, "Invalid");
  ASSERT_EQ(a, b);
}

TEST(Status, EqualsCopy) {
  const Status a(::mediapipe::StatusCode::kInvalidArgument, "Invalid");
  const Status b = a;
  ASSERT_EQ(a, b);
}

TEST(Status, EqualsDifferentCode) {
  const Status a(::mediapipe::StatusCode::kInvalidArgument, "Invalid");
  const Status b(::mediapipe::StatusCode::kInternal, "Internal");
  ASSERT_NE(a, b);
}

TEST(Status, EqualsDifferentMessage) {
  const Status a(::mediapipe::StatusCode::kInvalidArgument, "message");
  const Status b(::mediapipe::StatusCode::kInvalidArgument, "another");
  ASSERT_NE(a, b);
}

}  // namespace mediapipe
