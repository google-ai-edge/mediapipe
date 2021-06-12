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

#include <functional>
#include <list>
#include <memory>
#include <vector>

#include "absl/base/macros.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_context_manager.h"
#include "mediapipe/framework/input_stream_handler.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/tag_map.h"
#include "mediapipe/framework/tool/tag_map_helper.h"

namespace mediapipe {

namespace {

class BarrierInputStreamHandlerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    packet_type_.Set<std::string>();
    headers_ready_callback_ = [this]() {
      BarrierInputStreamHandlerTest::HeadersReadyNoOp();
    };
    notification_callback_ = [this]() {
      BarrierInputStreamHandlerTest::NotifyNoOp();
    };
    schedule_callback_ = std::bind(&BarrierInputStreamHandlerTest::Schedule,
                                   this, std::placeholders::_1);
    error_callback_ = std::bind(&BarrierInputStreamHandlerTest::RecordError,
                                this, std::placeholders::_1);
    setup_shards_callback_ =
        std::bind(&BarrierInputStreamHandlerTest::SetupShardsNoOp, this,
                  std::placeholders::_1);
    queue_full_callback_ =
        std::bind(&BarrierInputStreamHandlerTest::ReportQueueNoOp, this,
                  std::placeholders::_1, std::placeholders::_2);
    queue_not_full_callback_ =
        std::bind(&BarrierInputStreamHandlerTest::ReportQueueNoOp, this,
                  std::placeholders::_1, std::placeholders::_2);

    std::shared_ptr<tool::TagMap> input_tag_map =
        tool::CreateTagMap({"input_a", "input_b", "input_c"}).value();

    input_stream_managers_.reset(
        new InputStreamManager[input_tag_map->NumEntries()]);
    const std::vector<std::string>& names = input_tag_map->Names();
    for (CollectionItemId id = input_tag_map->BeginId();
         id < input_tag_map->EndId(); ++id) {
      const std::string& stream_name = names[id.value()];
      name_to_id_[stream_name] = id;
      MP_ASSERT_OK(input_stream_managers_[id.value()].Initialize(
          stream_name, &packet_type_, /*back_edge=*/false));
    }
    SetupInputStreamHandler(input_tag_map);
  }

  void SetupInputStreamHandler(
      const std::shared_ptr<tool::TagMap>& input_tag_map) {
    calculator_state_ = absl::make_unique<CalculatorState>(
        "Node", /*node_id=*/0, "Calculator", CalculatorGraphConfig::Node(),
        nullptr);
    calculator_context_manager_.Initialize(
        calculator_state_.get(), input_tag_map,
        /*output_tag_map=*/tool::CreateTagMap({"output_a"}).value(),
        /*calculator_run_in_parallel=*/false);

    absl::StatusOr<std::unique_ptr<mediapipe::InputStreamHandler>>
        status_or_handler = InputStreamHandlerRegistry::CreateByName(
            "BarrierInputStreamHandler", input_tag_map,
            &calculator_context_manager_, MediaPipeOptions(),
            /*calculator_run_in_parallel=*/false);
    ASSERT_TRUE(status_or_handler.ok());
    input_stream_handler_ = std::move(status_or_handler.value());
    MP_ASSERT_OK(input_stream_handler_->InitializeInputStreamManagers(
        input_stream_managers_.get()));
    MP_ASSERT_OK(
        calculator_context_manager_.PrepareForRun(setup_shards_callback_));
    input_stream_handler_->PrepareForRun(headers_ready_callback_,
                                         notification_callback_,
                                         schedule_callback_, error_callback_);
    input_stream_handler_->SetQueueSizeCallbacks(queue_full_callback_,
                                                 queue_not_full_callback_);
  }

  void HeadersReadyNoOp() {}

  void NotifyNoOp() {}

  void Schedule(CalculatorContext* calculator_context) {
    CHECK(calculator_context);
    calculator_context_ = calculator_context;
  }

  void RecordError(const absl::Status& error) { errors_.push_back(error); }

  absl::Status SetupShardsNoOp(CalculatorContext* calculator_context) {
    return absl::OkStatus();
  }

  void ReportQueueNoOp(InputStreamManager* stream, bool* stream_was_full) {}

  PacketType packet_type_;
  std::function<void()> headers_ready_callback_;
  std::function<void()> notification_callback_;
  std::function<void(CalculatorContext*)> schedule_callback_;
  std::function<void(absl::Status)> error_callback_;
  std::function<absl::Status(CalculatorContext*)> setup_shards_callback_;
  InputStreamManager::QueueSizeCallback queue_full_callback_;
  InputStreamManager::QueueSizeCallback queue_not_full_callback_;

  // Vector of errors encountered while using the stream.
  std::vector<absl::Status> errors_;

  std::unique_ptr<CalculatorState> calculator_state_;
  CalculatorContextManager calculator_context_manager_;
  CalculatorContext* calculator_context_;
  std::map<std::string, CollectionItemId> name_to_id_;
  std::unique_ptr<InputStreamHandler> input_stream_handler_;
  std::unique_ptr<InputStreamManager[]> input_stream_managers_;
};

// This test simulates how CalculatorNode::ProcessNode() uses an
// input stream handler and the associated input streams.
TEST_F(BarrierInputStreamHandlerTest, SimulateProcessNode) {
  Timestamp min_stream_timestamp;
  ASSERT_FALSE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));

  std::list<Packet> packets;
  packets.push_back(Adopt(new std::string("packet 1")).At(Timestamp(10)));
  packets.push_back(Adopt(new std::string("packet 2")).At(Timestamp(30)));
  packets.push_back(Adopt(new std::string("packet 3")).At(Timestamp(20)));
  input_stream_handler_->AddPackets(name_to_id_["input_a"], packets);
  ASSERT_FALSE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));

  packets.clear();
  packets.push_back(Adopt(new std::string("packet 1")).At(Timestamp(0)));
  input_stream_handler_->AddPackets(name_to_id_["input_b"], packets);
  ASSERT_FALSE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));

  packets.clear();
  packets.push_back(Adopt(new std::string("packet 1")).At(Timestamp(0)));
  packets.push_back(Adopt(new std::string("packet 2")).At(Timestamp(0)));
  input_stream_handler_->AddPackets(name_to_id_["input_c"], packets);
  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  for (const auto& name_and_id : name_to_id_) {
    const InputStream& input_stream =
        calculator_context_->Inputs().Get(name_and_id.second);
    EXPECT_FALSE(input_stream.Value().IsEmpty());
    EXPECT_EQ(input_stream.Value().Get<std::string>(), "packet 1");
  }

  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(calculator_context_->InputTimestamp(),
                                          &calculator_context_->Inputs());
  input_stream_handler_->ClearCurrentInputs(calculator_context_);
  for (const auto& name_and_id : name_to_id_) {
    const InputStream& input_stream =
        calculator_context_->Inputs().Get(name_and_id.second);
    EXPECT_TRUE(input_stream.Value().IsEmpty());
  }

  packets.clear();
  packets.push_back(Adopt(new std::string("packet 3")).At(Timestamp(0)));
  input_stream_handler_->AddPackets(name_to_id_["input_c"], packets);

  ASSERT_FALSE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));

  packets.clear();
  packets.push_back(Adopt(new std::string("packet 2")).At(Timestamp(1)));
  input_stream_handler_->AddPackets(name_to_id_["input_b"], packets);
  input_stream_handler_->SetNextTimestampBound(name_to_id_["input_b"],
                                               Timestamp::Done());

  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  for (const auto& name_and_id : name_to_id_) {
    const InputStream& input_stream =
        calculator_context_->Inputs().Get(name_and_id.second);
    EXPECT_FALSE(input_stream.Value().IsEmpty());
    EXPECT_EQ(input_stream.Value().Get<std::string>(), "packet 2");
  }

  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(calculator_context_->InputTimestamp(),
                                          &calculator_context_->Inputs());
  input_stream_handler_->ClearCurrentInputs(calculator_context_);

  for (const auto& name_and_id : name_to_id_) {
    const InputStream& input_stream =
        calculator_context_->Inputs().Get(name_and_id.second);
    EXPECT_TRUE(input_stream.Value().IsEmpty());
  }

  EXPECT_FALSE(
      input_stream_handler_->GetInputStreamManager(name_to_id_["input_a"])
          ->IsEmpty());
  EXPECT_TRUE(
      input_stream_handler_->GetInputStreamManager(name_to_id_["input_b"])
          ->IsEmpty());
  EXPECT_FALSE(
      input_stream_handler_->GetInputStreamManager(name_to_id_["input_c"])
          ->IsEmpty());

  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(Timestamp::Done(), calculator_context_->InputTimestamp());
  // For Close(), input set should be empty
  for (const auto& name_and_id : name_to_id_) {
    const InputStream& input_stream =
        calculator_context_->Inputs().Get(name_and_id.second);
    EXPECT_TRUE(input_stream.Value().IsEmpty());
  }

  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(calculator_context_->InputTimestamp(),
                                          &calculator_context_->Inputs());
  for (const auto& name_and_id : name_to_id_) {
    const InputStream& input_stream =
        calculator_context_->Inputs().Get(name_and_id.second);
    EXPECT_TRUE(input_stream.Value().IsEmpty());
  }

  EXPECT_TRUE(errors_.empty());
}

}  // namespace
}  // namespace mediapipe
