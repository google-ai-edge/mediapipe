// Copyright 2024 The MediaPipe Authors.
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

#include "mediapipe/framework/tool/graph_runtime_info_logger.h"

#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::tool {
namespace {

TEST(GraphRuntimeInfoLoggerTest, ShouldCaptureRuntimeInfo) {
  mediapipe::GraphRuntimeInfoConfig config;
  config.set_enable_graph_runtime_info(true);

  absl::Notification callback_called;
  GraphRuntimeInfoLogger logger;
  MP_ASSERT_OK(logger.StartInBackground(config, [&]() {
    callback_called.Notify();
    return GraphRuntimeInfo();
  }));
  EXPECT_TRUE(
      callback_called.WaitForNotificationWithTimeout(absl::Seconds(10)));
}

}  // namespace
}  // namespace mediapipe::tool
