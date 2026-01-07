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

#include "mediapipe/framework/input_stream_manager.h"

#include <memory>

#include "absl/memory/memory.h"
#include "mediapipe/framework/input_stream_shard.h"
#include "mediapipe/framework/lifetime_tracker.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {
class InputStreamManagerTest : public ::testing::Test {
 protected:
  InputStreamManagerTest() {}

  void SetUp() override {
    queue_becomes_full_count_ = 0;
    expected_queue_becomes_full_count_ = 0;
    queue_becomes_not_full_count_ = 0;
    expected_queue_becomes_not_full_count_ = 0;
    notify_ = false;
    num_packets_dropped_ = 0;
    popped_packet_ = Packet();
    stream_is_done_ = false;

    packet_type_.Set<std::string>();
    input_stream_manager_ = absl::make_unique<InputStreamManager>();
    MP_ASSERT_OK(input_stream_manager_->Initialize("a_test", &packet_type_,
                                                   /*back_edge=*/false));

    queue_full_callback_ =
        std::bind(&InputStreamManagerTest::ReportQueueBecomesFull, this,
                  std::placeholders::_1, std::placeholders::_2);
    queue_not_full_callback_ =
        std::bind(&InputStreamManagerTest::ReportQueueBecomesNotFull, this,
                  std::placeholders::_1, std::placeholders::_2);
    input_stream_manager_->PrepareForRun();
    input_stream_manager_->SetQueueSizeCallbacks(queue_full_callback_,
                                                 queue_not_full_callback_);
  }

  void TearDown() override {
    EXPECT_EQ(expected_queue_becomes_full_count_, queue_becomes_full_count_);
    EXPECT_EQ(expected_queue_becomes_not_full_count_,
              queue_becomes_not_full_count_);
  }

  void ReportQueueBecomesFull(InputStreamManager* stream,
                              bool* stream_was_full) {
    ++queue_becomes_full_count_;
  }

  void ReportQueueBecomesNotFull(InputStreamManager* stream,
                                 bool* stream_was_full) {
    ++queue_becomes_not_full_count_;
  }

  PacketType packet_type_;
  std::unique_ptr<InputStreamManager> input_stream_manager_;
  InputStreamManager::QueueSizeCallback queue_full_callback_;
  InputStreamManager::QueueSizeCallback queue_not_full_callback_;

  int expected_queue_becomes_full_count_;
  int expected_queue_becomes_not_full_count_;
  bool notify_;
  int num_packets_dropped_;
  Packet popped_packet_;
  bool stream_is_done_;

 private:
  int queue_becomes_full_count_;
  int queue_becomes_not_full_count_;
};

TEST_F(InputStreamManagerTest, Init) {}

TEST_F(InputStreamManagerTest, AddPackets) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(20)));
  packets.push_back(MakePacket<std::string>("packet 3").At(Timestamp(30)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_TRUE(notify_);
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  // After AddPackets(), the original packets should still be non-empty.
  for (const Packet& orignial_packet : packets) {
    EXPECT_FALSE(orignial_packet.IsEmpty());
  }
}

TEST_F(InputStreamManagerTest, MovePackets) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(20)));
  packets.push_back(MakePacket<std::string>("packet 3").At(Timestamp(30)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->MovePackets(&packets, &notify_));  // Notification
  EXPECT_TRUE(notify_);
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  // After MovePackets(), the original packets should become empty.
  for (const Packet& orignial_packet : packets) {
    EXPECT_TRUE(orignial_packet.IsEmpty());
  }
}

// InputStreamManager should reject the four timestamps that are not allowed in
// a stream: Timestamp::Unset(), Timestamp::Unstarted(),
// Timestamp::OneOverPostStream(), and Timestamp::Done().
TEST_F(InputStreamManagerTest, AddPacketUnset) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp::Unset()));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  absl::Status result =
      input_stream_manager_->AddPackets(packets, &notify_);  // No notification
  ASSERT_THAT(result.message(), testing::HasSubstr("Timestamp::Unset()"));
  EXPECT_FALSE(notify_);
}

TEST_F(InputStreamManagerTest, AddPacketUnstarted) {
  std::list<Packet> packets;
  packets.push_back(
      MakePacket<std::string>("packet 1").At(Timestamp::Unstarted()));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  absl::Status result =
      input_stream_manager_->AddPackets(packets, &notify_);  // No notification
  ASSERT_THAT(result.message(), testing::HasSubstr("Timestamp::Unstarted()"));
  EXPECT_FALSE(notify_);
}

TEST_F(InputStreamManagerTest, AddPacketOneOverPostStream) {
  std::list<Packet> packets;
  packets.push_back(
      MakePacket<std::string>("packet 1").At(Timestamp::OneOverPostStream()));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  absl::Status result =
      input_stream_manager_->AddPackets(packets, &notify_);  // No notification
  ASSERT_THAT(result.message(),
              testing::HasSubstr("Timestamp::OneOverPostStream()"));
  EXPECT_FALSE(notify_);
}

TEST_F(InputStreamManagerTest, AddPacketDone) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp::Done()));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  absl::Status result =
      input_stream_manager_->AddPackets(packets, &notify_);  // No notification
  ASSERT_THAT(result.message(), testing::HasSubstr("Timestamp::Done()"));
  EXPECT_FALSE(notify_);
}

TEST_F(InputStreamManagerTest, AddPacketsOnlyPreStream) {
  std::list<Packet> packets;
  packets.push_back(
      MakePacket<std::string>("packet 1").At(Timestamp::PreStream()));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);
}

// An attempt to add a packet after Timestamp::PreStream() should be rejected
// because the next timestamp bound is Timestamp::OneOverPostStream().
TEST_F(InputStreamManagerTest, AddPacketsAfterPreStream) {
  std::list<Packet> packets;
  packets.push_back(
      MakePacket<std::string>("packet 1").At(Timestamp::PreStream()));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(10)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  absl::Status result =
      input_stream_manager_->AddPackets(packets, &notify_);  // No notification
  ASSERT_THAT(result.message(),
              testing::HasSubstr("Timestamp::OneOverPostStream()"));
  EXPECT_FALSE(notify_);
}

TEST_F(InputStreamManagerTest, AddPacketsOnlyPostStream) {
  std::list<Packet> packets;
  packets.push_back(
      MakePacket<std::string>("packet 1").At(Timestamp::PostStream()));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);
}

// A packet at Timestamp::PostStream() must be the only Packet in an input
// stream.
TEST_F(InputStreamManagerTest, AddPacketsBeforePostStream) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(
      MakePacket<std::string>("packet 2").At(Timestamp::PostStream()));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  absl::Status result =
      input_stream_manager_->AddPackets(packets, &notify_);  // No notification
  ASSERT_THAT(result.message(), testing::HasSubstr("Timestamp::PostStream()"));
  EXPECT_FALSE(notify_);
}

TEST_F(InputStreamManagerTest, AddPacketsReverseTimestamps) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(20)));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(10)));
  packets.push_back(MakePacket<std::string>("packet 3").At(Timestamp(30)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  absl::Status result =
      input_stream_manager_->AddPackets(packets, &notify_);  // No notification
  ASSERT_THAT(result.message(),
              testing::HasSubstr(
                  "Current minimum expected timestamp is 21 but received 10"));
  EXPECT_FALSE(notify_);
}

TEST_F(InputStreamManagerTest, PopPacketAtTimestamp) {
  std::string expected_value_at_10("packet 1");
  std::string expected_value_at_20("packet 2");
  std::string expected_value_at_30("packet 3");
  std::list<Packet> packets;
  packets.push_back(
      MakePacket<std::string>(expected_value_at_10).At(Timestamp(10)));
  packets.push_back(
      MakePacket<std::string>(expected_value_at_20).At(Timestamp(20)));
  packets.push_back(
      MakePacket<std::string>(expected_value_at_30).At(Timestamp(30)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(input_stream_manager_->QueueHead().IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);
  EXPECT_EQ(expected_value_at_10,
            input_stream_manager_->QueueHead().Get<std::string>());
  EXPECT_EQ(Timestamp(10), input_stream_manager_->QueueHead().Timestamp());

  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(5), &num_packets_dropped_, &stream_is_done_);
  EXPECT_EQ(0, num_packets_dropped_);
  EXPECT_TRUE(popped_packet_.IsEmpty());
  EXPECT_FALSE(stream_is_done_);
  EXPECT_EQ(expected_value_at_10,
            input_stream_manager_->QueueHead().Get<std::string>());
  EXPECT_EQ(Timestamp(10), input_stream_manager_->QueueHead().Timestamp());

  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(10), &num_packets_dropped_, &stream_is_done_);
  EXPECT_EQ(0, num_packets_dropped_);
  ASSERT_FALSE(popped_packet_.IsEmpty());
  EXPECT_FALSE(stream_is_done_);
  EXPECT_EQ(expected_value_at_10, popped_packet_.Get<std::string>());
  EXPECT_EQ(Timestamp(10), popped_packet_.Timestamp());
  EXPECT_EQ(expected_value_at_20,
            input_stream_manager_->QueueHead().Get<std::string>());
  EXPECT_EQ(Timestamp(20), input_stream_manager_->QueueHead().Timestamp());

  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(15), &num_packets_dropped_, &stream_is_done_);
  EXPECT_EQ(0, num_packets_dropped_);
  EXPECT_TRUE(popped_packet_.IsEmpty());
  EXPECT_FALSE(stream_is_done_);
  EXPECT_EQ(expected_value_at_20,
            input_stream_manager_->QueueHead().Get<std::string>());
  EXPECT_EQ(Timestamp(20), input_stream_manager_->QueueHead().Timestamp());

  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(30), &num_packets_dropped_, &stream_is_done_);

  EXPECT_EQ(1, num_packets_dropped_);
  ASSERT_FALSE(popped_packet_.IsEmpty());
  EXPECT_FALSE(stream_is_done_);
  EXPECT_EQ(expected_value_at_30, popped_packet_.Get<std::string>());
  EXPECT_EQ(Timestamp(30), popped_packet_.Timestamp());

  EXPECT_TRUE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(input_stream_manager_->QueueHead().IsEmpty());

  notify_ = false;
  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  stream_is_done_ = false;
  MP_ASSERT_OK(input_stream_manager_->SetNextTimestampBound(Timestamp::Done(),
                                                            &notify_));
  EXPECT_TRUE(notify_);
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(40), &num_packets_dropped_, &stream_is_done_);
  EXPECT_EQ(0, num_packets_dropped_);
  EXPECT_TRUE(popped_packet_.IsEmpty());
  // Next timestamp bound reaches Timestamp::Done().
  EXPECT_TRUE(stream_is_done_);
}

TEST_F(InputStreamManagerTest, PopQueueHead) {
  input_stream_manager_->DisableTimestamps();
  std::string expected_value_at_10("packet 1");
  std::string expected_value_at_20("packet 2");
  std::string expected_value_at_30("packet 3");
  std::list<Packet> packets;
  packets.push_back(
      MakePacket<std::string>(expected_value_at_10).At(Timestamp(10)));
  packets.push_back(
      MakePacket<std::string>(expected_value_at_20).At(Timestamp(20)));
  packets.push_back(
      MakePacket<std::string>(expected_value_at_30).At(Timestamp(30)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(input_stream_manager_->QueueHead().IsEmpty());
  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_TRUE(notify_);
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_EQ(expected_value_at_10,
            input_stream_manager_->QueueHead().Get<std::string>());
  EXPECT_EQ(Timestamp(10), input_stream_manager_->QueueHead().Timestamp());

  popped_packet_ = input_stream_manager_->PopQueueHead(&stream_is_done_);
  ASSERT_FALSE(popped_packet_.IsEmpty());
  EXPECT_FALSE(stream_is_done_);
  EXPECT_EQ(expected_value_at_10, popped_packet_.Get<std::string>());
  EXPECT_EQ(Timestamp(10), popped_packet_.Timestamp());
  EXPECT_EQ(expected_value_at_20,
            input_stream_manager_->QueueHead().Get<std::string>());
  EXPECT_EQ(Timestamp(20), input_stream_manager_->QueueHead().Timestamp());

  stream_is_done_ = false;
  popped_packet_ = Packet();
  popped_packet_ = input_stream_manager_->PopQueueHead(&stream_is_done_);
  ASSERT_FALSE(popped_packet_.IsEmpty());
  EXPECT_FALSE(stream_is_done_);
  EXPECT_EQ(expected_value_at_20, popped_packet_.Get<std::string>());
  EXPECT_EQ(Timestamp(20), popped_packet_.Timestamp());
  EXPECT_EQ(expected_value_at_30,
            input_stream_manager_->QueueHead().Get<std::string>());
  EXPECT_EQ(Timestamp(30), input_stream_manager_->QueueHead().Timestamp());

  stream_is_done_ = false;
  popped_packet_ = Packet();
  popped_packet_ = input_stream_manager_->PopQueueHead(&stream_is_done_);
  ASSERT_FALSE(popped_packet_.IsEmpty());
  EXPECT_FALSE(stream_is_done_);
  EXPECT_EQ(expected_value_at_30, popped_packet_.Get<std::string>());
  EXPECT_EQ(Timestamp(30), popped_packet_.Timestamp());
  EXPECT_TRUE(input_stream_manager_->QueueHead().IsEmpty());

  notify_ = false;
  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  stream_is_done_ = false;
  MP_ASSERT_OK(input_stream_manager_->SetNextTimestampBound(Timestamp::Done(),
                                                            &notify_));
  EXPECT_TRUE(notify_);
  popped_packet_ = input_stream_manager_->PopQueueHead(&stream_is_done_);
  EXPECT_TRUE(popped_packet_.IsEmpty());
  // Next timestamp bound reaches Timestamp::Done().
  EXPECT_TRUE(stream_is_done_);
}

TEST_F(InputStreamManagerTest, BadPacketType) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<int>(10).At(Timestamp(10)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  absl::Status result =
      input_stream_manager_->AddPackets(packets, &notify_);  // No notification
  ASSERT_THAT(result.message(), testing::HasSubstr("Packet type mismatch"));
  EXPECT_FALSE(notify_);
}

TEST_F(InputStreamManagerTest, Close) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(20)));
  packets.push_back(MakePacket<std::string>("packet 3").At(Timestamp(30)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);

  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(40), &num_packets_dropped_, &stream_is_done_);
  // Dropped packets at timestamp 10, 20, and 30.
  EXPECT_EQ(3, num_packets_dropped_);
  ASSERT_TRUE(input_stream_manager_->IsEmpty());
  EXPECT_FALSE(stream_is_done_);

  input_stream_manager_->Close();
  EXPECT_TRUE(input_stream_manager_->IsEmpty());
}

TEST_F(InputStreamManagerTest, ReuseInputStreamManager) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(20)));
  packets.push_back(MakePacket<std::string>("packet 3").At(Timestamp(30)));

  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);

  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(40), &num_packets_dropped_, &stream_is_done_);
  // Dropped packets at timestamp 10, 20, and 30.
  EXPECT_EQ(3, num_packets_dropped_);
  ASSERT_TRUE(input_stream_manager_->IsEmpty());
  EXPECT_FALSE(stream_is_done_);

  input_stream_manager_->Close();
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  // Another run.
  input_stream_manager_->PrepareForRun();
  input_stream_manager_->SetQueueSizeCallbacks(queue_full_callback_,
                                               queue_not_full_callback_);
  notify_ = false;
  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  packets.clear();

  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(20)));
  packets.push_back(MakePacket<std::string>("packet 3").At(Timestamp(30)));

  EXPECT_TRUE(input_stream_manager_->IsEmpty());
  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);

  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(40), &num_packets_dropped_, &stream_is_done_);
  // Dropped packets at timestamp 10, 20, and 30.
  EXPECT_EQ(3, num_packets_dropped_);
  ASSERT_TRUE(input_stream_manager_->IsEmpty());
  EXPECT_FALSE(stream_is_done_);

  input_stream_manager_->Close();
  EXPECT_TRUE(input_stream_manager_->IsEmpty());
}

TEST_F(InputStreamManagerTest, MultipleNotifications) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(20)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);

  notify_ = false;
  packets.clear();
  packets.push_back(MakePacket<std::string>("packet 3").At(Timestamp(30)));
  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // No notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  // Notification isn't triggered since the queue is already non-empty.
  EXPECT_FALSE(notify_);

  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(50), &num_packets_dropped_, &stream_is_done_);
  EXPECT_EQ(3, num_packets_dropped_);
  ASSERT_TRUE(input_stream_manager_->IsEmpty());
  EXPECT_FALSE(stream_is_done_);

  notify_ = false;
  packets.clear();
  packets.push_back(MakePacket<std::string>("packet 4").At(Timestamp(60)));
  packets.push_back(MakePacket<std::string>("packet 5").At(Timestamp(70)));
  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);
}

TEST_F(InputStreamManagerTest, SetHeader) {
  Packet header = MakePacket<std::string>("blah");
  MP_ASSERT_OK(input_stream_manager_->SetHeader(header));

  EXPECT_EQ(header.Get<std::string>(),
            input_stream_manager_->Header().Get<std::string>());
  EXPECT_EQ(header.Timestamp(), input_stream_manager_->Header().Timestamp());
}

TEST_F(InputStreamManagerTest, BackwardsInTime) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(20)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);

  notify_ = false;
  MP_ASSERT_OK(input_stream_manager_->SetNextTimestampBound(
      Timestamp(50), &notify_));  // No notification
  // The queue is already non-empty.
  EXPECT_FALSE(notify_);

  notify_ = false;
  absl::Status result = input_stream_manager_->SetNextTimestampBound(
      Timestamp(40), &notify_);  // Set Timestamp bound backwards in time.
  ASSERT_THAT(result.message(), testing::HasSubstr("40"));
  ASSERT_THAT(result.message(), testing::HasSubstr("50"));
  // The queue is already non-empty.
  EXPECT_FALSE(notify_);

  notify_ = false;
  packets.clear();
  packets.push_back(MakePacket<std::string>("packet 3")
                        .At(Timestamp(30)));  // Backwards in time
  absl::Status result2 =
      input_stream_manager_->AddPackets(packets, &notify_);  // No notification
  ASSERT_THAT(result2.message(), testing::HasSubstr("50"));
  ASSERT_THAT(result2.message(), testing::HasSubstr("30"));
  EXPECT_FALSE(notify_);

  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(100), &num_packets_dropped_, &stream_is_done_);
  // Dropped packets at timestamp 10 and 20.
  EXPECT_EQ(2, num_packets_dropped_);
  EXPECT_FALSE(stream_is_done_);

  notify_ = false;
  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  packets.clear();
  packets.push_back(MakePacket<std::string>("packet 4").At(Timestamp(110)));
  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(150), &num_packets_dropped_, &stream_is_done_);
  // Dropped packet at timestamp 110.
  EXPECT_EQ(1, num_packets_dropped_);
  EXPECT_FALSE(stream_is_done_);

  notify_ = false;
  packets.clear();
  packets.push_back(MakePacket<std::string>("packet 5")
                        .At(Timestamp(130)));  // Backwards in time.
  absl::Status result3 =
      input_stream_manager_->AddPackets(packets, &notify_);  // No notification
  ASSERT_THAT(result3.message(), testing::HasSubstr("151"));
  ASSERT_THAT(result3.message(), testing::HasSubstr("130"));
  EXPECT_FALSE(notify_);
}

TEST_F(InputStreamManagerTest, SelectBackwardsInTime) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(20)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);

  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(15), &num_packets_dropped_, &stream_is_done_);
  EXPECT_EQ(1, num_packets_dropped_);
  EXPECT_TRUE(popped_packet_.IsEmpty());
  EXPECT_FALSE(stream_is_done_);

  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  EXPECT_DEATH(input_stream_manager_->PopPacketAtTimestamp(
                   Timestamp(14), &num_packets_dropped_, &stream_is_done_),
               "");

  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(100), &num_packets_dropped_, &stream_is_done_);
  EXPECT_EQ(1, num_packets_dropped_);
  EXPECT_FALSE(stream_is_done_);

  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(100), &num_packets_dropped_,
      &stream_is_done_);  // Same timestamp is allowed.
  EXPECT_FALSE(stream_is_done_);

  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  EXPECT_DEATH(input_stream_manager_->PopPacketAtTimestamp(
                   Timestamp(99), &num_packets_dropped_, &stream_is_done_),
               "");
}

TEST_F(InputStreamManagerTest, TimestampBound) {
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(20)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);

  bool is_empty = false;
  EXPECT_EQ(Timestamp(10),
            input_stream_manager_->MinTimestampOrBound(&is_empty));
  notify_ = false;
  MP_ASSERT_OK(input_stream_manager_->SetNextTimestampBound(
      Timestamp(30), &notify_));  // No notification.
  EXPECT_FALSE(notify_);
  EXPECT_EQ(Timestamp(10),
            input_stream_manager_->MinTimestampOrBound(&is_empty));
  notify_ = false;
  MP_ASSERT_OK(input_stream_manager_->SetNextTimestampBound(
      Timestamp(40), &notify_));  // No notification.
  EXPECT_FALSE(notify_);
  EXPECT_EQ(Timestamp(10),
            input_stream_manager_->MinTimestampOrBound(&is_empty));
  notify_ = false;
  MP_ASSERT_OK(input_stream_manager_->SetNextTimestampBound(
      Timestamp(50), &notify_));  // No notification.
  EXPECT_FALSE(notify_);

  EXPECT_EQ(Timestamp(10),
            input_stream_manager_->MinTimestampOrBound(&is_empty));
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(10), &num_packets_dropped_, &stream_is_done_);
  ASSERT_FALSE(popped_packet_.IsEmpty());
  EXPECT_EQ(Timestamp(10), popped_packet_.Timestamp());
  EXPECT_EQ(0, num_packets_dropped_);
  EXPECT_FALSE(stream_is_done_);

  EXPECT_EQ(Timestamp(20),
            input_stream_manager_->MinTimestampOrBound(&is_empty));
  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(20), &num_packets_dropped_, &stream_is_done_);
  ASSERT_FALSE(popped_packet_.IsEmpty());
  EXPECT_EQ(Timestamp(20), popped_packet_.Timestamp());
  EXPECT_EQ(0, num_packets_dropped_);
  EXPECT_FALSE(stream_is_done_);

  EXPECT_EQ(Timestamp(50),
            input_stream_manager_->MinTimestampOrBound(&is_empty));
  num_packets_dropped_ = 0;
  popped_packet_ = Packet();
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(50), &num_packets_dropped_, &stream_is_done_);
  EXPECT_TRUE(popped_packet_.IsEmpty());
  EXPECT_EQ(0, num_packets_dropped_);
  EXPECT_FALSE(stream_is_done_);

  // TODO These notifications may be bad if they schedule a
  // Calculator Process() call at times that are irrelevant.
  notify_ = false;
  MP_ASSERT_OK(input_stream_manager_->SetNextTimestampBound(
      Timestamp(60), &notify_));  // Notification.
  EXPECT_TRUE(notify_);
  EXPECT_EQ(Timestamp(60),
            input_stream_manager_->MinTimestampOrBound(&is_empty));
  notify_ = false;
  MP_ASSERT_OK(input_stream_manager_->SetNextTimestampBound(
      Timestamp(70), &notify_));  // Notification.
  EXPECT_TRUE(notify_);
  EXPECT_EQ(Timestamp(70),
            input_stream_manager_->MinTimestampOrBound(&is_empty));
  notify_ = false;
  MP_ASSERT_OK(input_stream_manager_->SetNextTimestampBound(
      Timestamp(80), &notify_));  // Notification.
  EXPECT_TRUE(notify_);
  EXPECT_EQ(Timestamp(80),
            input_stream_manager_->MinTimestampOrBound(&is_empty));

  notify_ = false;
  packets.clear();
  packets.push_back(MakePacket<std::string>("packet 3").At(Timestamp(90)));
  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_TRUE(notify_);

  EXPECT_EQ(Timestamp(90),
            input_stream_manager_->MinTimestampOrBound(&is_empty));
  num_packets_dropped_ = 0;
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(90), &num_packets_dropped_, &stream_is_done_);
  ASSERT_FALSE(popped_packet_.IsEmpty());
  EXPECT_EQ(Timestamp(90), popped_packet_.Timestamp());
  EXPECT_EQ(0, num_packets_dropped_);
  EXPECT_FALSE(stream_is_done_);

  EXPECT_EQ(Timestamp(91),
            input_stream_manager_->MinTimestampOrBound(&is_empty));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());
  input_stream_manager_->Close();
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  // Maybe this value should be left as undefined.
  EXPECT_EQ(Timestamp::Done(),
            input_stream_manager_->MinTimestampOrBound(&is_empty));
}

TEST_F(InputStreamManagerTest, QueueSizeTest) {
  std::list<Packet> packets;
  int max_queue_size = 2;
  input_stream_manager_->SetMaxQueueSize(max_queue_size);
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(20)));
  packets.push_back(MakePacket<std::string>("packet 3").At(Timestamp(30)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);

  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(50), &num_packets_dropped_, &stream_is_done_);
  EXPECT_TRUE(popped_packet_.IsEmpty());
  // Dropped packets at timestamp 10, 20, and 30.
  EXPECT_EQ(3, num_packets_dropped_);
  EXPECT_TRUE(input_stream_manager_->IsEmpty());
  EXPECT_FALSE(stream_is_done_);
  EXPECT_EQ(3, input_stream_manager_->NumPacketsAdded());

  packets.clear();
  packets.push_back(MakePacket<std::string>("packet 4").At(Timestamp(60)));
  packets.push_back(MakePacket<std::string>("packet 5").At(Timestamp(70)));
  notify_ = false;
  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);
  EXPECT_EQ(5, input_stream_manager_->NumPacketsAdded());

  expected_queue_becomes_full_count_ = 2;
  expected_queue_becomes_not_full_count_ = 1;
}

TEST_F(InputStreamManagerTest, InputReleaseTest) {
  packet_type_.Set<LifetimeTracker::Object>();
  input_stream_manager_ = absl::make_unique<InputStreamManager>();
  MP_ASSERT_OK(input_stream_manager_->Initialize("a_test", &packet_type_,
                                                 /*back_edge=*/false));
  input_stream_manager_->PrepareForRun();
  input_stream_manager_->SetQueueSizeCallbacks(queue_full_callback_,
                                               queue_not_full_callback_);

  LifetimeTracker tracker;
  Timestamp timestamp = Timestamp(0);
  auto new_packet = [&timestamp, &tracker] {
    return Adopt(tracker.MakeObject().release()).At(++timestamp);
  };

  input_stream_manager_->SetMaxQueueSize(3);
  MP_ASSERT_OK(input_stream_manager_->AddPackets({new_packet()}, &notify_));
  MP_ASSERT_OK(input_stream_manager_->AddPackets({new_packet()}, &notify_));
  MP_ASSERT_OK(input_stream_manager_->AddPackets({new_packet()}, &notify_));
  EXPECT_EQ(3, tracker.live_count());

  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(1), &num_packets_dropped_, &stream_is_done_);
  EXPECT_FALSE(stream_is_done_);
  EXPECT_EQ(3, tracker.live_count());
  popped_packet_ = Packet();
  EXPECT_EQ(2, tracker.live_count());

  num_packets_dropped_ = 0;
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(2), &num_packets_dropped_, &stream_is_done_);
  EXPECT_FALSE(stream_is_done_);
  EXPECT_EQ(2, tracker.live_count());
  popped_packet_ = Packet();
  EXPECT_EQ(1, tracker.live_count());

  num_packets_dropped_ = 0;
  popped_packet_ = input_stream_manager_->PopPacketAtTimestamp(
      Timestamp(3), &num_packets_dropped_, &stream_is_done_);
  EXPECT_FALSE(stream_is_done_);
  EXPECT_EQ(1, tracker.live_count());
  popped_packet_ = Packet();
  EXPECT_EQ(0, tracker.live_count());

  expected_queue_becomes_full_count_ = 1;
  expected_queue_becomes_not_full_count_ = 1;
}

// An attempt to add a packet after Timestamp::PreStream() should be allowed
// if packet timestamps don't need to be increasing.
TEST_F(InputStreamManagerTest, AddPacketsAfterPreStreamUntimed) {
  input_stream_manager_->DisableTimestamps();
  std::list<Packet> packets;
  packets.push_back(
      MakePacket<std::string>("packet 1").At(Timestamp::PreStream()));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(10)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);
}

// A packet at Timestamp::PostStream() doesn't need to be the only Packet in
// an input stream if packet timestamps don't need to be increasing.
TEST_F(InputStreamManagerTest, AddPacketsBeforePostStreamUntimed) {
  input_stream_manager_->DisableTimestamps();
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(
      MakePacket<std::string>("packet 2").At(Timestamp::PostStream()));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);
}

TEST_F(InputStreamManagerTest, BackwardsInTimeUntimed) {
  input_stream_manager_->DisableTimestamps();
  std::list<Packet> packets;
  packets.push_back(MakePacket<std::string>("packet 1").At(Timestamp(10)));
  packets.push_back(MakePacket<std::string>("packet 2").At(Timestamp(20)));
  EXPECT_TRUE(input_stream_manager_->IsEmpty());

  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_FALSE(input_stream_manager_->IsEmpty());
  EXPECT_TRUE(notify_);

  notify_ = false;
  MP_ASSERT_OK(input_stream_manager_->SetNextTimestampBound(
      Timestamp(50), &notify_));  // No notification
  EXPECT_FALSE(notify_);

  notify_ = false;
  MP_ASSERT_OK(input_stream_manager_->SetNextTimestampBound(
      Timestamp(40), &notify_));  // Set Timestamp bound backwards in time
  EXPECT_FALSE(notify_);

  packets.clear();
  packets.push_back(MakePacket<std::string>("packet 3")
                        .At(Timestamp(30)));  // Backwards in time
  notify_ = false;
  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // No notification
  // Notification isn't triggered since the queue is already non-empty.
  EXPECT_FALSE(notify_);

  // Consume the three packets in the input stream.

  input_stream_manager_->PopQueueHead(&stream_is_done_);
  EXPECT_FALSE(stream_is_done_);
  input_stream_manager_->PopQueueHead(&stream_is_done_);
  EXPECT_FALSE(stream_is_done_);
  input_stream_manager_->PopQueueHead(&stream_is_done_);
  EXPECT_FALSE(stream_is_done_);

  packets.clear();
  packets.push_back(MakePacket<std::string>("packet 4").At(Timestamp(110)));

  notify_ = false;
  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_TRUE(notify_);

  input_stream_manager_->PopQueueHead(&stream_is_done_);
  EXPECT_FALSE(stream_is_done_);

  packets.clear();
  packets.push_back(MakePacket<std::string>("packet 5")
                        .At(Timestamp(130)));  // Backwards in time

  notify_ = false;
  MP_ASSERT_OK(
      input_stream_manager_->AddPackets(packets, &notify_));  // Notification
  EXPECT_TRUE(notify_);
}

}  // namespace
}  // namespace mediapipe
