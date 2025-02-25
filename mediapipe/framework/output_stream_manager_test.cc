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

#include "mediapipe/framework/output_stream_manager.h"

#include <functional>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "mediapipe/framework/input_stream_handler.h"
#include "mediapipe/framework/input_stream_manager.h"
#include "mediapipe/framework/output_stream_shard.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/tag_map_helper.h"

namespace mediapipe {

namespace {

class OutputStreamManagerTest : public ::testing::Test {
 protected:
  OutputStreamManagerTest() = default;

  void SetUp() override {
    packet_type_.Set<std::string>();

    headers_ready_callback_ =
        std::bind(&OutputStreamManagerTest::HeadersReadyNoOp, this);
    notification_callback_ =
        std::bind(&OutputStreamManagerTest::NotifyNoOp, this);
    schedule_callback_ = std::bind(&OutputStreamManagerTest::ScheduleNoOp, this,
                                   std::placeholders::_1);
    error_callback_ = std::bind(&OutputStreamManagerTest::RecordError, this,
                                std::placeholders::_1);
    queue_full_callback_ =
        std::bind(&OutputStreamManagerTest::ReportQueueNoOp, this,
                  std::placeholders::_1, std::placeholders::_2);
    queue_not_full_callback_ =
        std::bind(&OutputStreamManagerTest::ReportQueueNoOp, this,
                  std::placeholders::_1, std::placeholders::_2);

    output_stream_manager_ = absl::make_unique<OutputStreamManager>();
    MP_ASSERT_OK(output_stream_manager_->Initialize("a_test", &packet_type_));
    output_stream_manager_->PrepareForRun(error_callback_);
    output_stream_shard_.SetSpec(output_stream_manager_->Spec());
    output_stream_manager_->ResetShard(&output_stream_shard_);

    std::shared_ptr<tool::TagMap> tag_map = tool::CreateTagMap(1).value();
    absl::StatusOr<std::unique_ptr<mediapipe::InputStreamHandler>>
        status_or_handler = InputStreamHandlerRegistry::CreateByName(
            "DefaultInputStreamHandler", tag_map, /*cc_manager=*/nullptr,
            MediaPipeOptions(), /*calculator_run_in_parallel=*/false);
    ASSERT_TRUE(status_or_handler.ok());
    input_stream_handler_ = std::move(status_or_handler.value());
    const CollectionItemId& id = tag_map->BeginId();

    MP_ASSERT_OK(input_stream_manager_.Initialize("a_test", &packet_type_,
                                                  /*back_edge=*/false));
    MP_ASSERT_OK(input_stream_handler_->InitializeInputStreamManagers(
        &input_stream_manager_));
    output_stream_manager_->AddMirror(input_stream_handler_.get(), id);
    input_stream_handler_->PrepareForRun(headers_ready_callback_,
                                         notification_callback_,
                                         schedule_callback_, error_callback_);
    input_stream_handler_->SetQueueSizeCallbacks(queue_full_callback_,
                                                 queue_not_full_callback_);
  }

  void HeadersReadyNoOp() {}

  void NotifyNoOp() {}

  void ScheduleNoOp(CalculatorContext* cc) {}

  void RecordError(const absl::Status& error) { errors_.push_back(error); }

  void ReportQueueNoOp(InputStreamManager* stream, bool* stream_was_full) {}

  // Returns the output_bound to verify.
  Timestamp ComputeBoundAndPropagateUpdates(Timestamp input_timestamp) {
    Timestamp output_bound =
        output_stream_manager_->ComputeOutputTimestampBound(
            output_stream_shard_, input_timestamp);
    output_stream_manager_->PropagateUpdatesToMirrors(output_bound,
                                                      &output_stream_shard_);
    return output_bound;
  }

  PacketType packet_type_;

  std::function<void()> headers_ready_callback_;
  std::function<void()> notification_callback_;
  std::function<void(CalculatorContext*)> schedule_callback_;
  std::function<void(absl::Status)> error_callback_;
  InputStreamManager::QueueSizeCallback queue_full_callback_;
  InputStreamManager::QueueSizeCallback queue_not_full_callback_;

  std::unique_ptr<OutputStreamManager> output_stream_manager_;
  OutputStreamShard output_stream_shard_;
  std::unique_ptr<InputStreamHandler> input_stream_handler_;
  InputStreamManager input_stream_manager_;

  // Vector of errors encountered while using the stream.
  std::vector<absl::Status> errors_;
};

TEST_F(OutputStreamManagerTest, Init) {}

TEST_F(OutputStreamManagerTest, ComputeOutputTimestampBoundWithoutOffset) {
  Timestamp input_timestamp = Timestamp(0);
  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // If the offset isn't enabled, the input timestamp and the output bound are
  // not related. Since the output stream shard is empty, the output bound is
  // still Timestamp::Unset().
  EXPECT_EQ(Timestamp::Unset(), output_bound);

  output_stream_shard_.AddPacket(
      MakePacket<std::string>("Packet 1").At(Timestamp(10)));
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output_bound is based on the timestamp of the last added packet.
  EXPECT_EQ(Timestamp(11), output_bound);

  output_stream_shard_.AddPacket(
      MakePacket<std::string>("Packet 2").At(Timestamp(20)));
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output_bound is based on the timestamp of the last added packet.
  EXPECT_EQ(Timestamp(21), output_bound);

  output_stream_shard_.SetNextTimestampBound(Timestamp(100));
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output_bound is based on the next timestamp bound.
  EXPECT_EQ(Timestamp(100), output_bound);

  input_timestamp = Timestamp::Max();
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // If the offset isn't enabled, the input timestamp and the output bound are
  // not related. The output bound is still based on the output stream shard.
  EXPECT_EQ(Timestamp(100), output_bound);
}

TEST_F(OutputStreamManagerTest, ComputeOutputTimestampBoundWithOffset) {
  output_stream_shard_.SetOffset(0);
  Timestamp input_timestamp = Timestamp(0);
  // If the OutputStreamShard is empty, the output_bound is the next allowed
  // timestamp of the input_timestamp plus the offset.
  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp(1), output_bound);

  output_stream_shard_.AddPacket(
      MakePacket<std::string>("Packet 1").At(Timestamp(10)));
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output_bound is based on the timestamp of the last added packet.
  EXPECT_EQ(Timestamp(11), output_bound);

  output_stream_shard_.AddPacket(
      MakePacket<std::string>("Packet 2").At(Timestamp(20)));
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output_bound is based on the timestamp of the last added packet.
  EXPECT_EQ(Timestamp(21), output_bound);

  output_stream_shard_.SetNextTimestampBound(Timestamp(100));
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output_bound is based on the next timestamp bound.
  EXPECT_EQ(Timestamp(100), output_bound);

  input_timestamp = Timestamp::Max();
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output_bound is based on the input timestamp, which is
  // Timestamp::Max().
  EXPECT_EQ(Timestamp::PostStream(), output_bound);

  input_timestamp = Timestamp::PostStream();
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output_bound is based on the input timestamp, which is
  // Timestamp::PostStream().
  EXPECT_EQ(Timestamp::OneOverPostStream(), output_bound);
}

TEST_F(OutputStreamManagerTest, ComputeOutputTimestampBoundWithNegativeOffset) {
  output_stream_shard_.SetOffset(-10);
  Timestamp input_timestamp = Timestamp(0);
  // If the OutputStreamShard is empty, the output_bound is the next allowed
  // timestamp of the input_timestamp plus the negative offset.
  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp(-9), output_bound);

  output_stream_shard_.AddPacket(
      MakePacket<std::string>("Packet 1").At(Timestamp(10)));
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output_bound is based on the timestamp of the last added packet.
  EXPECT_EQ(Timestamp(11), output_bound);

  output_stream_shard_.AddPacket(
      MakePacket<std::string>("Packet 2").At(Timestamp(20)));
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output_bound is based on the timestamp of the last added packet.
  EXPECT_EQ(Timestamp(21), output_bound);

  output_stream_shard_.SetNextTimestampBound(Timestamp(100));
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output_bound is based on the next timestamp bound.
  EXPECT_EQ(Timestamp(100), output_bound);

  input_timestamp = Timestamp::Max();
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output bound is the next allowed timestamp of Timestamp::Max() plus the
  // negative offset.
  Timestamp expcted_bound(Timestamp::Max().Value() - 9);
  EXPECT_EQ(expcted_bound, output_bound);

  input_timestamp = Timestamp::PostStream();
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output_bound is based on the input timestamp, which is
  // Timestamp::PostStream(). Even with negative offset, it's expected that no
  // futher timestamps will occur.
  EXPECT_EQ(Timestamp::OneOverPostStream(), output_bound);
}

TEST_F(OutputStreamManagerTest,
       ComputeOutputTimestampBoundWithoutOffsetAfterOpenNode) {
  Timestamp input_timestamp = Timestamp::Unstarted();
  // If the OutputStreamShard is empty, the output_bound is Timestamp::Unset().
  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp::Unset(), output_bound);

  output_stream_shard_.AddPacket(
      MakePacket<std::string>("Packet 1").At(Timestamp(20)));
  // The output_bound is based on the timestamp of the last added packet.
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp(21), output_bound);

  output_stream_shard_.Close();
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp::Done(), output_bound);
}

TEST_F(OutputStreamManagerTest,
       ComputeOutputTimestampBoundWithOffsetAfterOpenNode) {
  output_stream_shard_.SetOffset(0);
  Timestamp input_timestamp = Timestamp::Unstarted();
  // If the OutputStreamShard is empty, the output_bound is always
  // Timestamp::Unset() regardless of the offset.
  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp::Unset(), output_bound);

  output_stream_shard_.AddPacket(
      MakePacket<std::string>("Packet 1").At(Timestamp(20)));
  // The output_bound is based on the timestamp of the last added packet.
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp(21), output_bound);

  output_stream_shard_.Close();
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp::Done(), output_bound);
}

TEST_F(OutputStreamManagerTest,
       ComputeOutputTimestampBoundWithoutOffsetForPreStream) {
  Timestamp input_timestamp = Timestamp::PreStream();
  // If the OutputStreamShard is empty, the output bound is equal to the
  // Timestamp::Unset().
  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp::Unset(), output_bound);

  output_stream_shard_.AddPacket(
      MakePacket<std::string>("Packet 1").At(Timestamp(20)));
  // The output_bound is based on the timestamp of the last added packet.
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp(21), output_bound);

  output_stream_shard_.Close();
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp::Done(), output_bound);
}

TEST_F(OutputStreamManagerTest,
       ComputeOutputTimestampBoundWithOffsetForPreStream) {
  output_stream_shard_.SetOffset(0);
  Timestamp input_timestamp = Timestamp::PreStream();
  // If the OutputStreamShard is empty, the output_bound is always
  // Timestamp::Min() regardless of the offset.
  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp::Min(), output_bound);

  output_stream_shard_.AddPacket(
      MakePacket<std::string>("Packet 1").At(Timestamp(20)));
  // The output_bound is based on the timestamp of the last added packet.
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp(21), output_bound);

  output_stream_shard_.Close();
  output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp::Done(), output_bound);
}

TEST_F(OutputStreamManagerTest, AddPacket) {
  Timestamp input_timestamp = Timestamp(0);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp(10)));
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 2").At(Timestamp(20)));
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 3").At(Timestamp(30)));
  EXPECT_FALSE(output_stream_shard_.IsEmpty());
  EXPECT_TRUE(input_stream_manager_.IsEmpty());

  EXPECT_EQ(Timestamp(31), ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_TRUE(output_stream_shard_.IsEmpty());
  EXPECT_FALSE(input_stream_manager_.IsEmpty());
  EXPECT_TRUE(errors_.empty());
}

// The stream should reject the four timestamps that are not allowed in a
// stream: Timestamp::Unset(), Timestamp::Unstarted(),
// Timestamp::OneOverPostStream(), and Timestamp::Done().
TEST_F(OutputStreamManagerTest, AddPacketUnset) {
  Timestamp input_timestamp = Timestamp(0);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp::Unset()));
  // This error is reported by OutputStreamShard rather than InputStreamManager.
  ASSERT_EQ(1, errors_.size());
  EXPECT_TRUE(output_stream_shard_.IsEmpty());

  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output bound is still equal to Timestamp::Unset().
  EXPECT_EQ(Timestamp::Unset(), output_bound);

  output_stream_manager_->PropagateUpdatesToMirrors(output_bound,
                                                    &output_stream_shard_);
  ASSERT_EQ(1, errors_.size());
  EXPECT_THAT(errors_[0].ToString(), testing::HasSubstr("Timestamp::Unset()"));
  // Nothing changed in the input stream.
  EXPECT_TRUE(input_stream_manager_.IsEmpty());
}

TEST_F(OutputStreamManagerTest, AddPacketUnstarted) {
  Timestamp input_timestamp = Timestamp(0);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp::Unstarted()));
  // This error is reported by OutputStreamShard rather than InputStreamManager.
  ASSERT_EQ(1, errors_.size());
  EXPECT_TRUE(output_stream_shard_.IsEmpty());

  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output bound is still equal to Timestamp::Unset().
  EXPECT_EQ(Timestamp::Unset(), output_bound);

  output_stream_manager_->PropagateUpdatesToMirrors(output_bound,
                                                    &output_stream_shard_);
  ASSERT_EQ(1, errors_.size());
  EXPECT_THAT(errors_[0].ToString(),
              testing::HasSubstr("Timestamp::Unstarted()"));
  // Nothing changed in the input stream.
  EXPECT_TRUE(input_stream_manager_.IsEmpty());
}

TEST_F(OutputStreamManagerTest, AddPacketOneOverPostStream) {
  Timestamp input_timestamp = Timestamp(0);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp::OneOverPostStream()));
  // This error is reported by OutputStreamShard rather than InputStreamManager.
  ASSERT_EQ(1, errors_.size());
  EXPECT_TRUE(output_stream_shard_.IsEmpty());

  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output bound is still equal to Timestamp::Unset().
  EXPECT_EQ(Timestamp::Unset(), output_bound);

  output_stream_manager_->PropagateUpdatesToMirrors(output_bound,
                                                    &output_stream_shard_);
  ASSERT_EQ(1, errors_.size());
  EXPECT_THAT(errors_[0].ToString(),
              testing::HasSubstr("Timestamp::OneOverPostStream()"));
  // Nothing changed in the input stream.
  EXPECT_TRUE(input_stream_manager_.IsEmpty());
}

TEST_F(OutputStreamManagerTest, AddPacketdDone) {
  Timestamp input_timestamp = Timestamp(0);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp::Done()));
  // This error is reported by OutputStreamShard rather than InputStreamManager.
  ASSERT_EQ(1, errors_.size());
  EXPECT_TRUE(output_stream_shard_.IsEmpty());

  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // The output bound is still equal to Timestamp::Unset().
  EXPECT_EQ(Timestamp::Unset(), output_bound);

  output_stream_manager_->PropagateUpdatesToMirrors(output_bound,
                                                    &output_stream_shard_);
  ASSERT_EQ(1, errors_.size());
  EXPECT_THAT(errors_[0].ToString(), testing::HasSubstr("Timestamp::Done()"));
  // Nothing changed in the input stream.
  EXPECT_TRUE(input_stream_manager_.IsEmpty());
}

TEST_F(OutputStreamManagerTest, AddPacketOnlyPreStream) {
  Timestamp input_timestamp = Timestamp::PreStream();
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp::PreStream()));
  EXPECT_TRUE(input_stream_manager_.IsEmpty());

  EXPECT_EQ(Timestamp::OneOverPostStream(),
            ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_FALSE(input_stream_manager_.IsEmpty());
  EXPECT_TRUE(errors_.empty());
}

// An attempt to add a packet after Timestamp::PreStream() should be rejected
// because the next timestamp bound is Timestamp::OneOverPostStream().
TEST_F(OutputStreamManagerTest, AddPacketAfterPreStream) {
  Timestamp input_timestamp = Timestamp::PreStream();
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp::PreStream()));
  EXPECT_EQ(Timestamp::OneOverPostStream(),
            ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_FALSE(input_stream_manager_.IsEmpty());
  EXPECT_TRUE(errors_.empty());

  input_timestamp = Timestamp(10);
  output_stream_manager_->ResetShard(&output_stream_shard_);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 2").At(Timestamp(10)));
  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp(11), output_bound);
  // This error is reported by InputStreamManager rather than
  // OutputStreamManager.
  EXPECT_TRUE(errors_.empty());
  output_stream_manager_->PropagateUpdatesToMirrors(output_bound,
                                                    &output_stream_shard_);
  ASSERT_EQ(1, errors_.size());
  EXPECT_THAT(errors_[0].ToString(),
              testing::HasSubstr("Timestamp::OneOverPostStream()"));
}

TEST_F(OutputStreamManagerTest, AddPacketOnlyPostStream) {
  Timestamp input_timestamp = Timestamp::PostStream();
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp::PostStream()));
  EXPECT_TRUE(input_stream_manager_.IsEmpty());

  EXPECT_EQ(Timestamp::OneOverPostStream(),
            ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_FALSE(input_stream_manager_.IsEmpty());
  EXPECT_TRUE(errors_.empty());
}

// A packet at Timestamp::PostStream() must be the only Packet in a stream.
TEST_F(OutputStreamManagerTest, AddPacketBeforePostStream) {
  Timestamp input_timestamp = Timestamp(10);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp(10)));
  EXPECT_TRUE(input_stream_manager_.IsEmpty());
  EXPECT_EQ(Timestamp(11), ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_FALSE(input_stream_manager_.IsEmpty());
  EXPECT_TRUE(errors_.empty());

  input_timestamp = Timestamp(20);
  output_stream_manager_->ResetShard(&output_stream_shard_);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 2").At(Timestamp::PostStream()));
  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp::OneOverPostStream(), output_bound);
  // This error is reported by InputStreamManager rather than
  // OutputStreamManager.
  EXPECT_TRUE(errors_.empty());
  output_stream_manager_->PropagateUpdatesToMirrors(output_bound,
                                                    &output_stream_shard_);
  ASSERT_EQ(1, errors_.size());
  EXPECT_THAT(errors_[0].ToString(),
              testing::HasSubstr("Timestamp::PostStream()"));
}

TEST_F(OutputStreamManagerTest, AddPacketReverseTimestamps) {
  Timestamp input_timestamp = Timestamp(20);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp(20)));
  EXPECT_TRUE(input_stream_manager_.IsEmpty());
  EXPECT_EQ(Timestamp(21), ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_FALSE(input_stream_manager_.IsEmpty());
  EXPECT_TRUE(errors_.empty());

  input_timestamp = Timestamp(10);
  output_stream_manager_->ResetShard(&output_stream_shard_);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 2").At(Timestamp(10)));
  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  EXPECT_EQ(Timestamp(11), output_bound);
  // This error is reported by InputStreamManager rather than
  // OutputStreamManager.
  EXPECT_TRUE(errors_.empty());
  output_stream_manager_->PropagateUpdatesToMirrors(output_bound,
                                                    &output_stream_shard_);
  ASSERT_EQ(1, errors_.size());
  EXPECT_FALSE(errors_[0].ok());
}

TEST_F(OutputStreamManagerTest, SetNextTimestampBoundReverseTimestamps) {
  Timestamp input_timestamp = Timestamp(20);
  output_stream_shard_.SetNextTimestampBound(Timestamp(50));
  EXPECT_EQ(Timestamp(50), ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_TRUE(input_stream_manager_.IsEmpty());
  EXPECT_TRUE(errors_.empty());

  output_stream_manager_->ResetShard(&output_stream_shard_);
  output_stream_shard_.SetNextTimestampBound(
      Timestamp(40));  // Set Timestamp bound backwards in time.
  Timestamp next_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  // This error is reported by InputStreamManager rather than
  // OutputStreamManager.
  EXPECT_TRUE(errors_.empty());
  output_stream_manager_->PropagateUpdatesToMirrors(next_bound,
                                                    &output_stream_shard_);
  ASSERT_EQ(1, errors_.size());
  EXPECT_THAT(errors_[0].ToString(), testing::HasSubstr("50"));
  EXPECT_THAT(errors_[0].ToString(), testing::HasSubstr("40"));
}

TEST_F(OutputStreamManagerTest, BadPacketType) {
  output_stream_shard_.AddPacket(Adopt(new int(10)).At(Timestamp(10)));
  ASSERT_EQ(1, errors_.size());
  EXPECT_FALSE(errors_[0].ok());
}

TEST_F(OutputStreamManagerTest, SetHeader) {
  Packet header = MakePacket<std::string>("header");
  output_stream_shard_.SetHeader(header);
  output_stream_manager_->PropagateHeader();
  EXPECT_TRUE(errors_.empty());

  EXPECT_EQ(header.Get<std::string>(),
            output_stream_manager_->Header().Get<std::string>());
  EXPECT_EQ(header.Timestamp(), output_stream_manager_->Header().Timestamp());
  EXPECT_EQ(header.Get<std::string>(),
            input_stream_manager_.Header().Get<std::string>());
  EXPECT_EQ(header.Timestamp(), input_stream_manager_.Header().Timestamp());
}

TEST_F(OutputStreamManagerTest, SetHeaderWithTimestamp) {
  Packet header = MakePacket<std::string>("header").At(Timestamp(10));
  output_stream_shard_.SetHeader(header);
  output_stream_manager_->PropagateHeader();
  ASSERT_EQ(1, errors_.size());
  EXPECT_FALSE(errors_[0].ok());
}

// An attempt to add a packet after Timestamp::PreStream() should be allowed
// if packet timestamps don't need to be increasing.
TEST_F(OutputStreamManagerTest, AddPacketAfterPreStreamUntimed) {
  input_stream_manager_.DisableTimestamps();
  Timestamp input_timestamp = Timestamp::PreStream();
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp::PreStream()));
  EXPECT_EQ(Timestamp::OneOverPostStream(),
            ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_FALSE(input_stream_manager_.IsEmpty());
  EXPECT_TRUE(errors_.empty());

  input_timestamp = Timestamp(10);
  output_stream_manager_->ResetShard(&output_stream_shard_);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 2").At(Timestamp(10)));
  EXPECT_EQ(Timestamp(11), ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_TRUE(errors_.empty());
}

// A packet at Timestamp::PostStream() does not need to be the only Packet in
// a stream if packet timestamps don't need to be increasing.
TEST_F(OutputStreamManagerTest, AddPacketBeforePostStreamUntimed) {
  input_stream_manager_.DisableTimestamps();
  Timestamp input_timestamp = Timestamp(10);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp(10)));
  EXPECT_EQ(Timestamp(11), ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_FALSE(input_stream_manager_.IsEmpty());
  EXPECT_TRUE(errors_.empty());

  input_timestamp = Timestamp::PostStream();
  output_stream_manager_->ResetShard(&output_stream_shard_);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 2").At(Timestamp::PostStream()));
  EXPECT_EQ(Timestamp::OneOverPostStream(),
            ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_TRUE(errors_.empty());
}

TEST_F(OutputStreamManagerTest, AddPacketReverseTimestampsUntimed) {
  input_stream_manager_.DisableTimestamps();
  Timestamp input_timestamp = Timestamp(20);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp(20)));
  EXPECT_TRUE(input_stream_manager_.IsEmpty());
  EXPECT_EQ(Timestamp(21), ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_FALSE(input_stream_manager_.IsEmpty());
  EXPECT_TRUE(errors_.empty());

  input_timestamp = Timestamp(10);
  output_stream_manager_->ResetShard(&output_stream_shard_);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 2").At(Timestamp(10)));
  EXPECT_EQ(Timestamp(11), ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_TRUE(errors_.empty());
}

TEST_F(OutputStreamManagerTest, SetNextTimestampBoundReverseTimestampsUntimed) {
  input_stream_manager_.DisableTimestamps();
  Timestamp input_timestamp = Timestamp(20);
  output_stream_shard_.SetNextTimestampBound(Timestamp(50));
  EXPECT_EQ(Timestamp(50), ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_TRUE(errors_.empty());

  input_timestamp = Timestamp(30);
  output_stream_manager_->ResetShard(&output_stream_shard_);
  output_stream_shard_.SetNextTimestampBound(
      Timestamp(40));  // Set Timestamp bound backwards in time.
  EXPECT_EQ(Timestamp(40), ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_TRUE(errors_.empty());
}

TEST_F(OutputStreamManagerTest, AddPacketAndMovePacket) {
  Timestamp input_timestamp = Timestamp(10);
  Packet packet_1 = MakePacket<std::string>("packet 1").At(Timestamp(10));
  output_stream_shard_.AddPacket(packet_1);
  // packet_1 has an extra copy in the output stream.
  ASSERT_FALSE(packet_1.IsEmpty());
  MP_ASSERT_OK(packet_1.ValidateAsType<std::string>());
  EXPECT_EQ("packet 1", packet_1.Get<std::string>());

  Packet packet_2 = MakePacket<std::string>("packet 2").At(Timestamp(20));
  output_stream_shard_.AddPacket(std::move(packet_2));
  // packet_2 is moved into the output stream.
  ASSERT_TRUE(packet_2.IsEmpty());  // NOLINT used after std::move().

  EXPECT_TRUE(input_stream_manager_.IsEmpty());
  EXPECT_EQ(Timestamp(21), ComputeBoundAndPropagateUpdates(input_timestamp));
  EXPECT_FALSE(input_stream_manager_.IsEmpty());
  EXPECT_TRUE(errors_.empty());
}

TEST_F(OutputStreamManagerTest, ShouldReportNumPacketsAdded) {
  Timestamp input_timestamp = Timestamp(0);
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 1").At(Timestamp(1)));
  output_stream_shard_.AddPacket(
      MakePacket<std::string>("packet 2").At(Timestamp(2)));

  Timestamp output_bound = output_stream_manager_->ComputeOutputTimestampBound(
      output_stream_shard_, input_timestamp);
  output_stream_manager_->PropagateUpdatesToMirrors(output_bound,
                                                    &output_stream_shard_);
  ASSERT_TRUE(errors_.empty());
  EXPECT_EQ(input_stream_manager_.NumPacketsAdded(), 2);
  EXPECT_TRUE(output_stream_shard_.IsEmpty());
}

}  // namespace
}  // namespace mediapipe
