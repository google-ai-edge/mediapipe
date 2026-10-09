/* Copyright 2024 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "mediapipe/tasks/c/core/mp_logging.h"

#include <string>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace {

struct LogEntry {
  int severity;
  std::string message;
};

void TestLogCallback(int severity, const char* message, void* user_data) {
  auto* logs = static_cast<std::vector<LogEntry>*>(user_data);
  logs->push_back({severity, message});
}

TEST(MpLoggingTest, InterceptsAbseilLogs) {
  std::vector<LogEntry> logs;
  MpSetLoggingCallback(TestLogCallback, &logs);
  absl::Cleanup cleanup = [] { MpRemoveLoggingCallback(); };

  ABSL_LOG(INFO) << "Test info message";
  ABSL_LOG(WARNING) << "Test warning message";
  ABSL_LOG(ERROR) << "Test error message";

  // absl::LogSeverity::kInfo == 0, kWarning == 1, kError == 2
  ASSERT_GE(logs.size(), 3);

  // Note: other tests or initializations might log before our test runs,
  // so we check the last 3 messages.
  auto last_error = logs[logs.size() - 1];
  auto last_warning = logs[logs.size() - 2];
  auto last_info = logs[logs.size() - 3];

  EXPECT_EQ(last_info.severity, static_cast<int>(absl::LogSeverity::kInfo));
  EXPECT_THAT(last_info.message, testing::HasSubstr("Test info message"));

  EXPECT_EQ(last_warning.severity,
            static_cast<int>(absl::LogSeverity::kWarning));
  EXPECT_THAT(last_warning.message, testing::HasSubstr("Test warning message"));

  EXPECT_EQ(last_error.severity, static_cast<int>(absl::LogSeverity::kError));
  EXPECT_THAT(last_error.message, testing::HasSubstr("Test error message"));
}

}  // namespace
