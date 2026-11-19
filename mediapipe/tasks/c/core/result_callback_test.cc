// Copyright 2026 The MediaPipe Authors.
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

#include "mediapipe/tasks/c/core/result_callback.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/core/mp_status.h"

namespace mediapipe::tasks::c::core {
namespace {

struct TestResult {
  int value;
};

// The arguments of the last call of a callback. It is passed to the callback
// as `user_data`.
struct Call {
  int count = 0;
  MpStatus status = kMpOk;
  // The error message, or nullopt if `error_msg` was null.
  std::optional<std::string> error_msg;
  const TestResult* result = nullptr;
  int64_t timestamp_ms = 0;
};

void RecordCall(void* user_data, MpStatus status, const char* error_msg,
                const TestResult* result, int64_t timestamp_ms) {
  Call* call = reinterpret_cast<Call*>(user_data);
  ASSERT_NE(call, nullptr);
  ++call->count;
  call->status = status;
  call->error_msg = error_msg != nullptr ? std::optional<std::string>(error_msg)
                                         : std::nullopt;
  call->result = result;
  call->timestamp_ms = timestamp_ms;
}

int null_user_data_calls = 0;

void CountCallWithoutUserData(void* user_data, MpStatus status,
                              const char* error_msg, const TestResult* result) {
  if (user_data == nullptr && status == kMpOk && error_msg == nullptr &&
      result != nullptr) {
    ++null_user_data_calls;
  }
}

TEST(ResultCallbackTest, InvokeWithResultPassesUserDataAndResult) {
  Call call;
  const TestResult result = {.value = 1};
  ResultCallback callback(&RecordCall, &call);

  callback.InvokeWithResult(&result, /*timestamp_ms=*/42);

  EXPECT_EQ(call.count, 1);
  EXPECT_EQ(call.status, kMpOk);
  EXPECT_EQ(call.error_msg, std::nullopt);
  EXPECT_EQ(call.result, &result);
  EXPECT_EQ(call.timestamp_ms, 42);
}

TEST(ResultCallbackTest, InvokeWithErrorPassesStatusAndMessage) {
  Call call;
  ResultCallback callback(&RecordCall, &call);

  callback.InvokeWithError(absl::InvalidArgumentError("Invalid image."),
                           /*timestamp_ms=*/42);

  EXPECT_EQ(call.count, 1);
  EXPECT_EQ(call.status, kMpInvalidArgument);
  EXPECT_EQ(call.error_msg, "Invalid image.");
  EXPECT_EQ(call.result, nullptr);
  EXPECT_EQ(call.timestamp_ms, 42);
}

TEST(ResultCallbackTest, InvokeWithErrorPassesEmptyMessage) {
  Call call;
  ResultCallback callback(&RecordCall, &call);

  callback.InvokeWithError(absl::InternalError(""), /*timestamp_ms=*/0);

  EXPECT_EQ(call.status, kMpInternal);
  EXPECT_EQ(call.error_msg, "");
}

TEST(ResultCallbackTest, CopiesReportToTheSameUserData) {
  Call call;
  const TestResult result = {.value = 1};
  ResultCallback callback(&RecordCall, &call);
  ResultCallback copy = callback;

  callback.InvokeWithResult(&result, /*timestamp_ms=*/1);
  copy.InvokeWithResult(&result, /*timestamp_ms=*/2);

  EXPECT_EQ(call.count, 2);
  EXPECT_EQ(call.timestamp_ms, 2);
}

TEST(ResultCallbackTest, PassesNullUserData) {
  null_user_data_calls = 0;
  const TestResult result = {.value = 1};
  ResultCallback callback(&CountCallWithoutUserData, /*user_data=*/nullptr);

  callback.InvokeWithResult(&result);

  EXPECT_EQ(null_user_data_calls, 1);
}

}  // namespace
}  // namespace mediapipe::tasks::c::core
