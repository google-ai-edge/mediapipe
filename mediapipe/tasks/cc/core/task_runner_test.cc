/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/core/task_runner.h"

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"

namespace mediapipe {
namespace tasks {
namespace core {
namespace {

CalculatorGraphConfig GetPassThroughGraphConfig() {
  return ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        input_stream: "in"
        output_stream: "out"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "in"
          output_stream: "out"
        })pb");
}

// A calculator to generate runtime errors.
class ErrorCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Index(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    return absl::InternalError("An intended error for testing");
  }
};
REGISTER_CALCULATOR(ErrorCalculator);

CalculatorGraphConfig GetErrorCalculatorGraphConfig() {
  return ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        input_stream: "in"
        output_stream: "out"
        node {
          calculator: "ErrorCalculator"
          input_stream: "in"
          output_stream: "out"
        })pb");
}

CalculatorGraphConfig GetModelSidePacketsToStreamPacketsGraphConfig(
    const std::string& model_resources_tag) {
  return ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
      R"(
    input_stream: "tick"
    output_stream: "model_out"
    output_stream: "metadata_extractor_out"
    node {
      calculator: "ModelResourcesCalculator"
      output_side_packet: "MODEL:model"
      output_side_packet: "METADATA_EXTRACTOR:metadata_extractor"
      options {
        [mediapipe.tasks.ModelResourcesCalculatorOptions.ext] {
          model_resources_tag: "$0"
        }
      }
    }
    node {
      calculator: "SidePacketToStreamCalculator"
      input_stream: "TICK:tick"
      input_side_packet: "model"
      output_stream: "AT_TICK:model_out"
    }
    node {
      calculator: "SidePacketToStreamCalculator"
      input_stream: "TICK:tick"
      input_side_packet: "metadata_extractor"
      output_stream: "AT_TICK:metadata_extractor_out"
    })",
      /*$0=*/model_resources_tag));
}

}  // namespace

class TaskRunnerTest : public tflite_shims::testing::Test {};

TEST_F(TaskRunnerTest, ConfigWithNoOutputStream) {
  CalculatorGraphConfig proto = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: 'in'
    node {
      calculator: 'RandomCalculator'
      input_stream: 'INPUT:in'
      output_stream: 'OUTPUT:output'
    })pb");
  auto status_or_runner = TaskRunner::Create(proto);
  ASSERT_FALSE(status_or_runner.ok());
  ASSERT_THAT(status_or_runner.status().message(),
              testing::HasSubstr("no valid output streams"));
}

TEST_F(TaskRunnerTest, RunnerIsNotRunning) {
  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          TaskRunner::Create(GetPassThroughGraphConfig()));

  MP_ASSERT_OK(runner->Close());
  // Sends input data to runner after it gets closed.
  auto status = runner->Send({{"in", MakePacket<int>(0)}});
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.message(), testing::HasSubstr("not running"));
}

TEST_F(TaskRunnerTest, WrongProcessingMode) {
  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          TaskRunner::Create(GetPassThroughGraphConfig()));
  auto status = runner->Send({{"in", MakePacket<int>(0)}});
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.message(),
              testing::HasSubstr("the result callback is not provided"));
  MP_ASSERT_OK(runner->Close());

  MP_ASSERT_OK_AND_ASSIGN(
      auto runner2,
      TaskRunner::Create(GetPassThroughGraphConfig(),
                         /*model_resources=*/nullptr,
                         [](absl::StatusOr<PacketMap> status_or_packets) {}));
  auto status_or_result = runner2->Process({{"in", MakePacket<int>(0)}});
  ASSERT_FALSE(status_or_result.ok());
  ASSERT_THAT(status_or_result.status().message(),
              testing::HasSubstr("the result callback is provided"));
  MP_ASSERT_OK(runner2->Close());
}

TEST_F(TaskRunnerTest, WrongTimestampOrderInSyncCalls) {
  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          TaskRunner::Create(GetPassThroughGraphConfig()));
  MP_ASSERT_OK_AND_ASSIGN(
      auto results,
      runner->Process({{"in", MakePacket<int>(1).At(Timestamp(1))}}));
  // Sends a packet with an earlier timestamp, which is not allowed.
  auto status_or_results =
      runner->Process({{"in", MakePacket<int>(0).At(Timestamp(0))}});
  ASSERT_FALSE(status_or_results.ok());
  ASSERT_THAT(status_or_results.status().message(),
              testing::HasSubstr("monotonically increasing"));
  // Sends a packet with timestamp 1 again, which is not allowed.
  status_or_results =
      runner->Process({{"in", MakePacket<int>(1).At(Timestamp(1))}});
  ASSERT_FALSE(status_or_results.ok());
  ASSERT_THAT(status_or_results.status().message(),
              testing::HasSubstr("monotonically increasing"));
  status_or_results =
      runner->Process({{"in", MakePacket<int>(2).At(Timestamp(2))}});
  MP_ASSERT_OK(status_or_results);
  MP_ASSERT_OK(runner->Close());
}

TEST_F(TaskRunnerTest, WrongTimestampOrderInAsyncCalls) {
  std::function<void(absl::StatusOr<PacketMap>)> callback(
      [](absl::StatusOr<PacketMap> status_or_packets) {
        ASSERT_TRUE(status_or_packets.ok());
        ASSERT_EQ(1, status_or_packets.value().size());
        Packet out_packet = status_or_packets.value()["out"];
        EXPECT_EQ(out_packet.Timestamp().Value(), out_packet.Get<int>());
      });
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, TaskRunner::Create(GetPassThroughGraphConfig(),
                                      /*model_resources=*/nullptr, callback));
  MP_ASSERT_OK(runner->Send({{"in", MakePacket<int>(1).At(Timestamp(1))}}));
  // Sends a packet with an earlier timestamp, which is not allowed.
  auto status = runner->Send({{"in", MakePacket<int>(0).At(Timestamp(0))}});
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.message(), testing::HasSubstr("monotonically increasing"));

  // Sends a packet with timestamp 1 again, which is not allowed.
  status = runner->Send({{"in", MakePacket<int>(1).At(Timestamp(1))}});
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.message(), testing::HasSubstr("monotonically increasing"));

  MP_ASSERT_OK(runner->Send({{"in", MakePacket<int>(2).At(Timestamp(2))}}));
  MP_ASSERT_OK(runner->Close());
}

TEST_F(TaskRunnerTest, OneThreadSyncAPICalls) {
  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          TaskRunner::Create(GetPassThroughGraphConfig()));
  // Calls Process() 100 times from the same thread.
  for (int i = 0; i < 100; ++i) {
    auto status_or_result = runner->Process({{"in", MakePacket<int>(i)}});
    ASSERT_TRUE(status_or_result.ok());
    EXPECT_EQ(i, status_or_result.value()["out"].Get<int>());
  }
  MP_ASSERT_OK(runner->Restart());
  for (int i = 0; i < 100; ++i) {
    auto status_or_result = runner->Process({{"in", MakePacket<int>(i)}});
    ASSERT_TRUE(status_or_result.ok());
    EXPECT_EQ(i, status_or_result.value()["out"].Get<int>());
  }
  MP_ASSERT_OK(runner->Close());
}

TEST_F(TaskRunnerTest, MultiThreadSyncAPICallsWithoutTimestamp) {
  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          TaskRunner::Create(GetPassThroughGraphConfig()));

  constexpr int kNumThreads = 10;
  std::atomic<int> active_worker_threads = 0;
  std::thread threads[kNumThreads];
  // Calls Process() in multiple threads simultaneously.
  for (int i = 0; i < kNumThreads; ++i) {
    ++active_worker_threads;
    std::thread([i, &runner, &active_worker_threads]() {
      for (int j = 0; j < 30; ++j) {
        auto status_or_result =
            runner->Process({{"in", MakePacket<int>(i * j)}});
        ASSERT_TRUE(status_or_result.ok());
        EXPECT_EQ(i * j, status_or_result.value()["out"].Get<int>());
      }
      --active_worker_threads;
    }).detach();
  }
  while (active_worker_threads) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  MP_ASSERT_OK(runner->Close());
}

TEST_F(TaskRunnerTest, AsyncAPICalls) {
  std::function<void(absl::StatusOr<PacketMap>)> callback(
      [](absl::StatusOr<PacketMap> status_or_packets) {
        ASSERT_TRUE(status_or_packets.ok());
        ASSERT_EQ(1, status_or_packets.value().size());
        Packet out_packet = status_or_packets.value()["out"];
        EXPECT_EQ(out_packet.Timestamp().Value(), out_packet.Get<int>());
      });
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, TaskRunner::Create(GetPassThroughGraphConfig(),
                                      /*model_resources=*/nullptr, callback));
  for (int i = 0; i < 100; ++i) {
    MP_ASSERT_OK(runner->Send({{"in", MakePacket<int>(i).At(Timestamp(i))}}));
  }
  MP_ASSERT_OK(runner->Restart());
  for (int i = 0; i < 100; ++i) {
    MP_ASSERT_OK(runner->Send({{"in", MakePacket<int>(i).At(Timestamp(i))}}));
  }
  MP_ASSERT_OK(runner->Close());
}

TEST_F(TaskRunnerTest, ReportErrorInSyncAPICall) {
  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          TaskRunner::Create(GetErrorCalculatorGraphConfig()));
  auto status_or_result = runner->Process({{"in", MakePacket<int>(0)}});
  ASSERT_FALSE(status_or_result.ok());
  ASSERT_THAT(status_or_result.status().message(),
              testing::HasSubstr("An intended error for testing"));
}

TEST_F(TaskRunnerTest, ReportErrorInAsyncAPICall) {
  std::function<void(absl::StatusOr<PacketMap>)> callback(
      [](absl::StatusOr<PacketMap> status_or_packets) {
        ASSERT_TRUE(status_or_packets.ok());
        ASSERT_EQ(1, status_or_packets.value().size());
        Packet out_packet = status_or_packets.value()["out"];
        EXPECT_EQ(out_packet.Timestamp().Value(), out_packet.Get<int>());
      });
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, TaskRunner::Create(GetErrorCalculatorGraphConfig(),
                                      /*model_resources=*/nullptr, callback));
  MP_ASSERT_OK(runner->Send({{"in", MakePacket<int>(0).At(Timestamp(0))}}));
  auto status = runner->Close();
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.message(),
              testing::HasSubstr("An intended error for testing"));
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
