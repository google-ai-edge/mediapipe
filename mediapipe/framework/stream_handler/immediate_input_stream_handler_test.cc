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

#include <functional>
#include <list>
#include <memory>
#include <vector>

#include "absl/base/macros.h"
#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_context_manager.h"
#include "mediapipe/framework/calculator_state.h"
#include "mediapipe/framework/input_stream_handler.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/tag_map.h"
#include "mediapipe/framework/tool/tag_map_helper.h"

namespace mediapipe {

namespace {

class ImmediateInputStreamHandlerTest : public ::testing::Test {
 protected:
  ImmediateInputStreamHandlerTest() {
    packet_type_.Set<std::string>();
    headers_ready_callback_ = [this]() {
      ImmediateInputStreamHandlerTest::HeadersReadyNoOp();
    };
    notification_callback_ = [this]() {
      ImmediateInputStreamHandlerTest::NotifyNoOp();
    };
    schedule_callback_ = std::bind(&ImmediateInputStreamHandlerTest::Schedule,
                                   this, std::placeholders::_1);
    error_callback_ = std::bind(&ImmediateInputStreamHandlerTest::RecordError,
                                this, std::placeholders::_1);
    setup_shards_callback_ =
        std::bind(&ImmediateInputStreamHandlerTest::SetupShardsNoOp, this,
                  std::placeholders::_1);
    queue_full_callback_ =
        std::bind(&ImmediateInputStreamHandlerTest::ReportQueueNoOp, this,
                  std::placeholders::_1, std::placeholders::_2);
    queue_not_full_callback_ =
        std::bind(&ImmediateInputStreamHandlerTest::ReportQueueNoOp, this,
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
      MEDIAPIPE_CHECK_OK(input_stream_managers_[id.value()].Initialize(
          stream_name, &packet_type_, /*back_edge=*/false));
    }
    SetupInputStreamHandler(input_tag_map);
  }

  void SetupInputStreamHandler(
      const std::shared_ptr<tool::TagMap>& input_tag_map) {
    calculator_state_ = std::make_unique<CalculatorState>(
        "Node", /*node_id=*/0, "Calculator", CalculatorGraphConfig::Node(),
        /*profiling_context=*/nullptr,
        /*graph_service_manager=*/nullptr);
    cc_manager_.Initialize(
        calculator_state_.get(), input_tag_map,
        /*output_tag_map=*/tool::CreateTagMap({"output_a"}).value(),
        /*calculator_run_in_parallel=*/false);

    absl::StatusOr<std::unique_ptr<mediapipe::InputStreamHandler>>
        status_or_handler = InputStreamHandlerRegistry::CreateByName(
            "ImmediateInputStreamHandler", input_tag_map, &cc_manager_,
            MediaPipeOptions(),
            /*calculator_run_in_parallel=*/false);
    ASSERT_TRUE(status_or_handler.ok());
    input_stream_handler_ = std::move(status_or_handler.value());
    MP_ASSERT_OK(input_stream_handler_->InitializeInputStreamManagers(
        input_stream_managers_.get()));
    MP_ASSERT_OK(cc_manager_.PrepareForRun(setup_shards_callback_));
    input_stream_handler_->PrepareForRun(headers_ready_callback_,
                                         notification_callback_,
                                         schedule_callback_, error_callback_);
    input_stream_handler_->SetQueueSizeCallbacks(queue_full_callback_,
                                                 queue_not_full_callback_);
  }

  void HeadersReadyNoOp() {}

  void NotifyNoOp() {}

  void Schedule(CalculatorContext* cc) {
    ABSL_CHECK(cc);
    cc_ = cc;
  }

  void RecordError(const absl::Status& error) { errors_.push_back(error); }

  absl::Status SetupShardsNoOp(CalculatorContext* calculator_context) {
    return absl::OkStatus();
  }

  void ReportQueueNoOp(InputStreamManager* stream, bool* stream_was_full) {}

  void ExpectPackets(
      const InputStreamShardSet& input_set,
      const std::map<std::string, std::string>& expected_values) {
    for (const auto& name_and_id : name_to_id_) {
      const InputStream& input_stream = input_set.Get(name_and_id.second);
      if (mediapipe::ContainsKey(expected_values, name_and_id.first)) {
        ASSERT_FALSE(input_stream.Value().IsEmpty());
        EXPECT_EQ(input_stream.Value().Get<std::string>(),
                  mediapipe::FindOrDie(expected_values, name_and_id.first));
      } else {
        EXPECT_TRUE(input_stream.Value().IsEmpty());
      }
    }
  }

  const InputStream& Input(const CollectionItemId& id) {
    ABSL_CHECK(cc_);
    return cc_->Inputs().Get(id);
  }

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
  CalculatorContextManager cc_manager_;
  CalculatorContext* cc_;
  std::map<std::string, CollectionItemId> name_to_id_;
  std::unique_ptr<InputStreamHandler> input_stream_handler_;
  std::unique_ptr<InputStreamManager[]> input_stream_managers_;
};

// This test checks that a node is not considered ready for Process() if no
// packets have arrived.
TEST_F(ImmediateInputStreamHandlerTest, EmptyPacketsNotReady) {
  Timestamp min_stream_timestamp;
  ASSERT_FALSE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
}

// This test checks that a node is considered ready for Process() if any of the
// input streams has a packet available.
TEST_F(ImmediateInputStreamHandlerTest, AnyPacketsReady) {
  Timestamp min_stream_timestamp;
  std::list<Packet> packets;
  packets.push_back(Adopt(new std::string("packet 1")).At(Timestamp(10)));
  input_stream_handler_->AddPackets(name_to_id_["input_a"], packets);
  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  ExpectPackets(cc_->Inputs(), {{"input_a", "packet 1"}});
}

// This test checks that a node is considered ready for Process() if any of the
// input streams has become done.
TEST_F(ImmediateInputStreamHandlerTest, StreamDoneReady) {
  Timestamp min_stream_timestamp;
  std::list<Packet> packets;

  // One packet arrives, ready for process.
  packets.push_back(Adopt(new std::string("packet 1")).At(Timestamp(10)));
  input_stream_handler_->AddPackets(name_to_id_["input_a"], packets);
  EXPECT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  ExpectPackets(cc_->Inputs(), {{"input_a", "packet 1"}});
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  // No packets arrive, not ready for process.
  EXPECT_FALSE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  ExpectPackets(cc_->Inputs(), {});

  // Timestamp::Done arrives, ready for process.
  input_stream_handler_->SetNextTimestampBound(name_to_id_["input_b"],
                                               Timestamp::Done());
  EXPECT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  ExpectPackets(cc_->Inputs(), {});
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  // No timestamp arrives, not ready for process.
  EXPECT_FALSE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  ExpectPackets(cc_->Inputs(), {});

  // Timestamp 200 arrives, not ready for process.
  input_stream_handler_->SetNextTimestampBound(name_to_id_["input_a"],
                                               Timestamp(200));
  EXPECT_FALSE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  ExpectPackets(cc_->Inputs(), {});

  // Another Timestamp::Done arrives, ready for process.
  input_stream_handler_->SetNextTimestampBound(name_to_id_["input_a"],
                                               Timestamp::Done());
  EXPECT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  ExpectPackets(cc_->Inputs(), {});
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);
}

// This test checks that the state is ReadyForClose after all streams reach
// Timestamp::Max.
TEST_F(ImmediateInputStreamHandlerTest, ReadyForCloseAfterTimestampMax) {
  Timestamp min_stream_timestamp;
  std::list<Packet> packets;

  // One packet arrives, ready for process.
  packets.push_back(Adopt(new std::string("packet 1")).At(Timestamp(10)));
  input_stream_handler_->AddPackets(name_to_id_["input_a"], packets);
  EXPECT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(Timestamp(10), cc_->InputTimestamp());
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  // No packets arrive, not ready.
  EXPECT_FALSE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(Timestamp::Unset(), cc_->InputTimestamp());

  // Timestamp::Max arrives, ready for close.
  input_stream_handler_->SetNextTimestampBound(
      name_to_id_["input_a"], Timestamp::Max().NextAllowedInStream());
  input_stream_handler_->SetNextTimestampBound(
      name_to_id_["input_b"], Timestamp::Max().NextAllowedInStream());
  input_stream_handler_->SetNextTimestampBound(
      name_to_id_["input_c"], Timestamp::Max().NextAllowedInStream());

  EXPECT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(Timestamp::Done(), cc_->InputTimestamp());
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);
}

// This test checks that when any stream is done, the state is ready to close.
TEST_F(ImmediateInputStreamHandlerTest, ReadyForClose) {
  Timestamp min_stream_timestamp;
  std::list<Packet> packets;
  packets.push_back(Adopt(new std::string("packet 1")).At(Timestamp(1)));
  input_stream_handler_->AddPackets(name_to_id_["input_b"], packets);
  input_stream_handler_->SetNextTimestampBound(name_to_id_["input_b"],
                                               Timestamp::Done());
  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  ExpectPackets(cc_->Inputs(), {{"input_b", "packet 1"}});
  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  EXPECT_TRUE(
      input_stream_handler_->GetInputStreamManager(name_to_id_["input_b"])
          ->IsEmpty());
  input_stream_handler_->SetNextTimestampBound(name_to_id_["input_a"],
                                               Timestamp::Done());
  input_stream_handler_->SetNextTimestampBound(name_to_id_["input_c"],
                                               Timestamp::Done());
  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(Timestamp::Done(), cc_->InputTimestamp());
  ExpectPackets(cc_->Inputs(), {});

  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  ExpectPackets(cc_->Inputs(), {});
  EXPECT_TRUE(errors_.empty());
}

TEST_F(ImmediateInputStreamHandlerTest, ProcessTimestampBounds) {
  input_stream_handler_->SetProcessTimestampBounds(true);

  Timestamp min_stream_timestamp;
  ASSERT_FALSE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::PreStream());

  const auto& input_a_id = name_to_id_["input_a"];
  const auto& input_b_id = name_to_id_["input_b"];
  const auto& input_c_id = name_to_id_["input_c"];

  std::list<Packet> packets;
  packets.push_back(Adopt(new std::string("packet 1")).At(Timestamp(1)));
  input_stream_handler_->AddPackets(input_b_id, packets);
  input_stream_handler_->SetNextTimestampBound(input_b_id, Timestamp::Done());

  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp(1));
  ExpectPackets(cc_->Inputs(), {{"input_b", "packet 1"}});
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp(1));
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Unstarted());
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Unstarted());

  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Unstarted());
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Unstarted());

  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  EXPECT_TRUE(
      input_stream_handler_->GetInputStreamManager(input_b_id)->IsEmpty());

  input_stream_handler_->SetNextTimestampBound(input_a_id, Timestamp::Done());
  input_stream_handler_->SetNextTimestampBound(input_c_id, Timestamp::Done());

  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Max());

  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);
  EXPECT_TRUE(errors_.empty());

  // Schedule invocation for Close.
  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp::Done());
  EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp::Unset());
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Unset());
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Unset());

  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);
  EXPECT_TRUE(errors_.empty());
}

TEST_F(ImmediateInputStreamHandlerTest,
       ProcessTimestampBoundsNoOpScheduleInvocations) {
  input_stream_handler_->SetProcessTimestampBounds(true);

  const auto& input_a_id = name_to_id_["input_a"];
  const auto& input_b_id = name_to_id_["input_b"];
  const auto& input_c_id = name_to_id_["input_c"];

  Timestamp min_stream_timestamp;
  std::list<Packet> packets;
  packets.push_back(Adopt(new std::string("packet 1")).At(Timestamp(1)));
  input_stream_handler_->AddPackets(input_b_id, packets);
  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp(1));
  ExpectPackets(cc_->Inputs(), {{"input_b", "packet 1"}});
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp(1));
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Unstarted());
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Unstarted());

  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  input_stream_handler_->SetNextTimestampBound(input_a_id, Timestamp::Done());
  input_stream_handler_->SetNextTimestampBound(input_c_id, Timestamp::Done());

  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp(1));
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Max());

  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);
  EXPECT_TRUE(errors_.empty());

  // Try to schedule invocations several times again. Considering nothing
  // changed since last invocation nothing should be scheduled.
  for (int i = 0; i < 3; ++i) {
    ASSERT_FALSE(input_stream_handler_->ScheduleInvocations(
        /*max_allowance=*/1, &min_stream_timestamp));
    EXPECT_EQ(min_stream_timestamp, Timestamp(2));
    EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
    EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp::Unset());
    EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
    EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Unset());
    EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
    EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Unset());
  }

  input_stream_handler_->SetNextTimestampBound(input_b_id, Timestamp::Done());

  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Max());

  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);
  EXPECT_TRUE(errors_.empty());

  // Schedule invocation for Close.
  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp::Done());
  EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp::Unset());
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Unset());
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Unset());

  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);
  EXPECT_TRUE(errors_.empty());

  // Try to schedule invocations several times again. Considering nothing
  // changed since last invocation nothing should be scheduled.
  for (int i = 0; i < 3; ++i) {
    ASSERT_FALSE(input_stream_handler_->ScheduleInvocations(
        /*max_allowance=*/1, &min_stream_timestamp));
    EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
    EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
    EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp::Unset());
    EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
    EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Unset());
    EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
    EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Unset());
  }
}

// Due to some temporary changes in ImmediateInputStreamHandler some packets
// - were queued but never released
// - were released in incorrect order
// As other test cases were passing, this test case is designed to ensure that.
TEST_F(ImmediateInputStreamHandlerTest, VerifyPacketsReleaseOrder) {
  input_stream_handler_->SetProcessTimestampBounds(true);

  const auto& input_a_id = name_to_id_["input_a"];
  const auto& input_b_id = name_to_id_["input_b"];
  const auto& input_c_id = name_to_id_["input_c"];

  Packet packet_a = Adopt(new std::string("packet a"));
  Packet packet_b = Adopt(new std::string("packet b"));
  Packet packet_c = Adopt(new std::string("packet c"));
  input_stream_handler_->AddPackets(input_a_id, {packet_a.At(Timestamp(1))});
  input_stream_handler_->AddPackets(input_b_id, {packet_b.At(Timestamp(2))});
  input_stream_handler_->AddPackets(input_c_id, {packet_c.At(Timestamp(3))});

  Timestamp min_stream_timestamp;
  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp(1));
  ASSERT_FALSE(Input(input_a_id).IsEmpty());
  EXPECT_EQ(Input(input_a_id).Get<std::string>(), "packet a");
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp(1));
  EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp(1));
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp(2));

  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  input_stream_handler_->AddPackets(input_a_id, {packet_a.At(Timestamp(5))});
  input_stream_handler_->AddPackets(input_b_id, {packet_b.At(Timestamp(5))});
  input_stream_handler_->AddPackets(input_c_id, {packet_c.At(Timestamp(5))});

  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp(2));
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp(4));
  ASSERT_FALSE(Input(input_b_id).IsEmpty());
  EXPECT_EQ(Input(input_b_id).Get<std::string>(), "packet b");
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp(2));
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp(2));

  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp(3));
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp(4));
  EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp(4));
  ASSERT_FALSE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Get<std::string>(), "packet c");
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp(3));

  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp(5));
  ASSERT_FALSE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Get<std::string>(), "packet a");
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp(5));
  ASSERT_FALSE(Input(input_b_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_b_id).Get<std::string>(), "packet b");
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp(5));
  ASSERT_FALSE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Get<std::string>(), "packet c");
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp(5));

  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  input_stream_handler_->SetNextTimestampBound(input_a_id, Timestamp::Done());
  input_stream_handler_->SetNextTimestampBound(input_b_id, Timestamp::Done());
  input_stream_handler_->SetNextTimestampBound(input_c_id, Timestamp::Done());

  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Max());
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Max());

  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  // Schedule invocation for Close.
  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_EQ(cc_->InputTimestamp(), Timestamp::Done());
  EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp::Unset());
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Unset());
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Unset());

  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  ASSERT_FALSE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  EXPECT_EQ(min_stream_timestamp, Timestamp::Unset());
  EXPECT_TRUE(Input(input_b_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_b_id).Value().Timestamp(), Timestamp::Unset());
  EXPECT_TRUE(Input(input_a_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_a_id).Value().Timestamp(), Timestamp::Unset());
  EXPECT_TRUE(Input(input_c_id).Value().IsEmpty());
  EXPECT_EQ(Input(input_c_id).Value().Timestamp(), Timestamp::Unset());
}

// This test simulates how CalculatorNode::ProcessNode() uses an input
// stream handler and the associated input streams.
TEST_F(ImmediateInputStreamHandlerTest, SimulateProcessNode) {
  Timestamp min_stream_timestamp;
  std::list<Packet> packets;
  packets.push_back(Adopt(new std::string("packet 1")).At(Timestamp(10)));
  packets.push_back(Adopt(new std::string("packet 2")).At(Timestamp(30)));
  packets.push_back(Adopt(new std::string("packet 3")).At(Timestamp(40)));
  input_stream_handler_->AddPackets(name_to_id_["input_a"], packets);

  packets.clear();
  packets.push_back(Adopt(new std::string("packet 1")).At(Timestamp(10)));
  input_stream_handler_->AddPackets(name_to_id_["input_b"], packets);

  packets.clear();
  packets.push_back(Adopt(new std::string("packet 1")).At(Timestamp(10)));
  packets.push_back(Adopt(new std::string("packet 2")).At(Timestamp(30)));
  input_stream_handler_->AddPackets(name_to_id_["input_c"], packets);
  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  ExpectPackets(cc_->Inputs(), {{"input_a", "packet 1"},
                                {"input_b", "packet 1"},
                                {"input_c", "packet 1"}});
  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  for (const auto& name_and_id : name_to_id_) {
    const InputStream& input_stream = cc_->Inputs().Get(name_and_id.second);
    EXPECT_TRUE(input_stream.Value().IsEmpty());
  }

  packets.clear();
  packets.push_back(Adopt(new std::string("packet 3")).At(Timestamp(40)));
  input_stream_handler_->AddPackets(name_to_id_["input_c"], packets);

  packets.clear();
  packets.push_back(Adopt(new std::string("packet 2")).At(Timestamp(30)));
  input_stream_handler_->AddPackets(name_to_id_["input_b"], packets);
  input_stream_handler_->SetNextTimestampBound(name_to_id_["input_b"],
                                               Timestamp::Done());
  ASSERT_TRUE(input_stream_handler_->ScheduleInvocations(
      /*max_allowance=*/1, &min_stream_timestamp));
  ExpectPackets(cc_->Inputs(), {{"input_a", "packet 2"},
                                {"input_b", "packet 2"},
                                {"input_c", "packet 2"}});
  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  ExpectPackets(cc_->Inputs(), {});

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
  ExpectPackets(cc_->Inputs(),
                {{"input_a", "packet 3"}, {"input_c", "packet 3"}});
  // FinalizeInputSet() is a no-op.
  input_stream_handler_->FinalizeInputSet(cc_->InputTimestamp(),
                                          &cc_->Inputs());
  input_stream_handler_->ClearCurrentInputs(cc_);

  ExpectPackets(cc_->Inputs(), {});

  EXPECT_TRUE(errors_.empty());
}

}  // namespace
}  // namespace mediapipe
