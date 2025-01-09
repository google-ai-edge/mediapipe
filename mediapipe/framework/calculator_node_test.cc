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

#include "mediapipe/framework/calculator_node.h"

#include <unistd.h>

#include <memory>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/output_side_packet_impl.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

class CountCalculator : public CalculatorBase {
 public:
  CountCalculator() { ++num_constructed_; }
  ~CountCalculator() override { ++num_destroyed_; }

  static absl::Status GetContract(CalculatorContract* cc) {
    ++num_fill_expectations_;
    cc->Inputs().Get(cc->Inputs().BeginId()).Set<int>();
    cc->Outputs().Get(cc->Outputs().BeginId()).Set<int>();
    cc->InputSidePackets().Get(cc->InputSidePackets().BeginId()).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    ++num_open_;
    // Simulate doing nontrivial work to ensure that the time spent in the
    // method will register on streamz each time it is called.
    usleep(100);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    ++num_process_;
    int input_stream_int = cc->Inputs().Get(cc->Inputs().BeginId()).Get<int>();
    int side_packet_int =
        cc->InputSidePackets().Get(cc->InputSidePackets().BeginId()).Get<int>();
    cc->Outputs()
        .Get(cc->Outputs().BeginId())
        .AddPacket(MakePacket<int>(input_stream_int + side_packet_int)
                       .At(cc->InputTimestamp()));
    // Simulate doing nontrivial work to ensure that the time spent in the
    // method will register on streamz each time it is called.
    usleep(100);
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    ++num_close_;
    // Simulate doing nontrivial work to ensure that the time spent in the
    // method will register on streamz each time it is called.
    usleep(100);
    return absl::OkStatus();
  }

  static int num_constructed_;
  static int num_fill_expectations_;
  static int num_open_;
  static int num_process_;
  static int num_close_;
  static int num_destroyed_;
};
REGISTER_CALCULATOR(CountCalculator);

int CountCalculator::num_constructed_ = 0;
int CountCalculator::num_fill_expectations_ = 0;
int CountCalculator::num_open_ = 0;
int CountCalculator::num_process_ = 0;
int CountCalculator::num_close_ = 0;
int CountCalculator::num_destroyed_ = 0;

void SourceNodeOpenedNoOp() {}

void CheckFail(const absl::Status& status) {
  ABSL_LOG(FATAL) << "The test triggered the error callback with status: "
                  << status;
}

class CalculatorNodeTest : public ::testing::Test {
 public:
  void ReadyForOpen(int* count) { ++(*count); }

  void Notification(CalculatorContext* cc, int* count) {
    ABSL_CHECK(cc);
    cc_ = cc;
    ++(*count);
  }

 protected:
  void InitializeEnvironment(bool use_tags) {
    CountCalculator::num_constructed_ = 0;
    CountCalculator::num_fill_expectations_ = 0;
    CountCalculator::num_open_ = 0;
    CountCalculator::num_process_ = 0;
    CountCalculator::num_close_ = 0;
    CountCalculator::num_destroyed_ = 0;

    std::string first_two_nodes_string =
        "node {\n"  // Node index 0
        "  calculator: \"SidePacketsToStreamsCalculator\"\n"
        "  input_side_packet: \"input_b\"\n"    // Input side packet index 0
        "  output_stream: \"unused_stream\"\n"  // Output stream 0
        "}\n"
        "node {\n"  // Node index 1
        "  calculator: \"PassThroughCalculator\"\n"
        "  input_stream: \"unused_stream\"\n"  // Input stream index 0
        "  output_stream: \"stream_a\"\n"      // Output stream index 1
        "  input_side_packet: \"input_a\"\n"   // Input side packet index 1
        "  input_side_packet: \"input_b\"\n"   // Input side packet index 2
        "}\n";
    CalculatorGraphConfig graph_config;
    // Add the test for the node under test.
    if (use_tags) {
      graph_config = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          first_two_nodes_string +
          "node {\n"  // Node index 2
          "  calculator: \"CountCalculator\"\n"
          "  input_stream: \"INPUT_TAG:stream_a\"\n"    // Input stream index 1
          "  output_stream: \"OUTPUT_TAG:stream_b\"\n"  // Output stream index 2
          // Input side packet index 3
          "  input_side_packet: \"INPUT_SIDE_PACKET_TAG:input_a\"\n"
          "}\n");
    } else {
      graph_config = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          first_two_nodes_string +
          "node {\n"  // Node index 2
          "  calculator: \"CountCalculator\"\n"
          "  input_stream: \"stream_a\"\n"      // Input stream index 1
          "  output_stream: \"stream_b\"\n"     // Output stream index 2
          "  input_side_packet: \"input_a\"\n"  // Input side packet index 3
          "}\n");
    }
    MEDIAPIPE_CHECK_OK(validated_graph_.Initialize(graph_config));
    MEDIAPIPE_CHECK_OK(InitializeStreams());

    input_side_packets_.emplace("input_a", Adopt(new int(42)));
    input_side_packets_.emplace("input_b", Adopt(new int(42)));

    node_ = std::make_unique<CalculatorNode>();
    MP_ASSERT_OK(node_->Initialize(
        &validated_graph_, {NodeTypeInfo::NodeType::CALCULATOR, 2},
        input_stream_managers_.get(), output_stream_managers_.get(),
        output_side_packets_.get(), &buffer_size_hint_, graph_profiler_,
        /*graph_service_manager=*/nullptr));
  }

  absl::Status PrepareNodeForRun() {
    return node_->PrepareForRun(                      //
        input_side_packets_,                          //
        service_packets_,                             //
        std::bind(&CalculatorNodeTest::ReadyForOpen,  //
                  this,                               //
                  &ready_for_open_count_),            //
        SourceNodeOpenedNoOp,                         //
        std::bind(&CalculatorNodeTest::Notification,  //
                  this, std::placeholders::_1,        //
                  &schedule_count_),                  //
        CheckFail,                                    //
        nullptr);
  }

  absl::Status InitializeStreams() {
    // START OF: code is copied from
    // CalculatorGraph::InitializePacketGeneratorGraph.
    // Create and initialize the output side packets.
    output_side_packets_ = std::make_unique<OutputSidePacketImpl[]>(
        validated_graph_.OutputSidePacketInfos().size());
    for (int index = 0; index < validated_graph_.OutputSidePacketInfos().size();
         ++index) {
      const EdgeInfo& edge_info =
          validated_graph_.OutputSidePacketInfos()[index];
      MP_RETURN_IF_ERROR(output_side_packets_[index].Initialize(
          edge_info.name, edge_info.packet_type));
    }
    // END OF: code is copied from
    // CalculatorGraph::InitializePacketGeneratorGraph.

    // START OF: code is copied from CalculatorGraph::InitializeStreams.
    // Create and initialize the input streams.
    input_stream_managers_.reset(
        new InputStreamManager[validated_graph_.InputStreamInfos().size()]);
    for (int index = 0; index < validated_graph_.InputStreamInfos().size();
         ++index) {
      const EdgeInfo& edge_info = validated_graph_.InputStreamInfos()[index];
      MP_RETURN_IF_ERROR(input_stream_managers_[index].Initialize(
          edge_info.name, edge_info.packet_type, edge_info.back_edge));
    }

    // Create and initialize the output streams.
    output_stream_managers_.reset(
        new OutputStreamManager[validated_graph_.OutputStreamInfos().size()]);
    for (int index = 0; index < validated_graph_.OutputStreamInfos().size();
         ++index) {
      const EdgeInfo& edge_info = validated_graph_.OutputStreamInfos()[index];
      MP_RETURN_IF_ERROR(output_stream_managers_[index].Initialize(
          edge_info.name, edge_info.packet_type));
    }
    // END OF: code is copied from CalculatorGraph::InitializeStreams.

    stream_a_manager_ = &output_stream_managers_[1];
    stream_b_manager_ = &output_stream_managers_[2];
    return absl::OkStatus();
  }

  virtual void SimulateParentOpenNode() { stream_a_manager_->LockIntroData(); }

  virtual void TestCleanupAfterRunTwice();

  std::map<std::string, Packet> input_side_packets_;
  std::map<std::string, Packet> service_packets_;

  std::unique_ptr<InputStreamManager[]> input_stream_managers_;
  std::unique_ptr<OutputStreamManager[]> output_stream_managers_;
  std::unique_ptr<OutputSidePacketImpl[]> output_side_packets_;

  // A pointer to the output stream manager for stream_a.
  // An alias for &output_stream_managers_[1].
  OutputStreamManager* stream_a_manager_;
  // A pointer to the output stream manager for stream_b.
  // An alias for &output_stream_managers_[2].
  OutputStreamManager* stream_b_manager_;

  std::unique_ptr<CalculatorNode> node_;

  ValidatedGraphConfig validated_graph_;
  std::shared_ptr<ProfilingContext> graph_profiler_ =
      std::make_shared<ProfilingContext>();

  int ready_for_open_count_ = 0;
  int schedule_count_ = 0;

  int buffer_size_hint_ = -1;
  // Stores the CalculatorContext passed to the ready_callback_ of node_, and we
  // pass this to node_->ProcessNode().
  CalculatorContext* cc_;
};

TEST_F(CalculatorNodeTest, Initialize) {
  InitializeEnvironment(/*use_tags=*/false);
  EXPECT_EQ(2, node_->Id());
  EXPECT_THAT(node_->DebugName(), ::testing::HasSubstr("CountCalculator"));

  EXPECT_FALSE(node_->Prepared());
  EXPECT_FALSE(node_->Opened());
  EXPECT_FALSE(node_->Closed());

  EXPECT_EQ(0, CountCalculator::num_constructed_);
  EXPECT_EQ(1, CountCalculator::num_fill_expectations_);
  EXPECT_EQ(0, CountCalculator::num_open_);
  EXPECT_EQ(0, CountCalculator::num_process_);
  EXPECT_EQ(0, CountCalculator::num_close_);
  EXPECT_EQ(0, CountCalculator::num_destroyed_);
}

TEST_F(CalculatorNodeTest, PrepareForRun) {
  InitializeEnvironment(/*use_tags=*/false);
  MP_ASSERT_OK(PrepareNodeForRun());

  EXPECT_TRUE(node_->Prepared());
  EXPECT_FALSE(node_->Opened());
  EXPECT_FALSE(node_->Closed());

  EXPECT_EQ(0, ready_for_open_count_);
  EXPECT_EQ(0, schedule_count_);

  EXPECT_EQ(1, CountCalculator::num_constructed_);
  EXPECT_EQ(1, CountCalculator::num_fill_expectations_);
  EXPECT_EQ(0, CountCalculator::num_open_);
  EXPECT_EQ(0, CountCalculator::num_process_);
  EXPECT_EQ(0, CountCalculator::num_close_);
  EXPECT_EQ(0, CountCalculator::num_destroyed_);
}

TEST_F(CalculatorNodeTest, Open) {
  InitializeEnvironment(/*use_tags=*/false);
  MP_ASSERT_OK(PrepareNodeForRun());

  EXPECT_EQ(0, ready_for_open_count_);
  SimulateParentOpenNode();
  MP_EXPECT_OK(node_->OpenNode());

  EXPECT_TRUE(node_->Prepared());
  EXPECT_TRUE(node_->Opened());
  EXPECT_FALSE(node_->Closed());

  // Nodes are not immediately scheduled upon opening.
  EXPECT_EQ(0, schedule_count_);

  EXPECT_EQ(1, CountCalculator::num_constructed_);
  EXPECT_EQ(1, CountCalculator::num_fill_expectations_);
  EXPECT_EQ(1, CountCalculator::num_open_);
  EXPECT_EQ(0, CountCalculator::num_process_);
  EXPECT_EQ(0, CountCalculator::num_close_);
  EXPECT_EQ(0, CountCalculator::num_destroyed_);
}

TEST_F(CalculatorNodeTest, Process) {
  InitializeEnvironment(/*use_tags=*/false);
  MP_ASSERT_OK(PrepareNodeForRun());

  SimulateParentOpenNode();
  MP_EXPECT_OK(node_->OpenNode());

  OutputStreamShard stream_a_shard;
  stream_a_shard.SetSpec(stream_a_manager_->Spec());
  stream_a_shard.Add(new int(1), Timestamp(1));
  stream_a_manager_->PropagateUpdatesToMirrors(Timestamp(2), &stream_a_shard);
  EXPECT_EQ(1, schedule_count_);
  // Expects that a CalculatorContext has been prepared.
  EXPECT_NE(nullptr, cc_);
  EXPECT_TRUE(node_->TryToBeginScheduling());
  MP_EXPECT_OK(node_->ProcessNode(cc_));

  cc_ = nullptr;
  node_->EndScheduling();
  EXPECT_EQ(1, schedule_count_);
  // Expects that no CalculatorContext is prepared by EndScheduling().
  EXPECT_EQ(nullptr, cc_);

  EXPECT_TRUE(node_->Prepared());
  EXPECT_TRUE(node_->Opened());
  EXPECT_FALSE(node_->Closed());

  EXPECT_EQ(1, schedule_count_);

  EXPECT_EQ(1, CountCalculator::num_constructed_);
  EXPECT_EQ(1, CountCalculator::num_fill_expectations_);
  EXPECT_EQ(1, CountCalculator::num_open_);
  EXPECT_EQ(1, CountCalculator::num_process_);
  EXPECT_EQ(0, CountCalculator::num_close_);
  EXPECT_EQ(0, CountCalculator::num_destroyed_);
}

TEST_F(CalculatorNodeTest, ProcessSeveral) {
  InitializeEnvironment(/*use_tags=*/false);
  MP_ASSERT_OK(PrepareNodeForRun());

  SimulateParentOpenNode();
  MP_EXPECT_OK(node_->OpenNode());

  OutputStreamShard stream_a_shard;
  stream_a_shard.SetSpec(stream_a_manager_->Spec());
  stream_a_shard.Add(new int(1), Timestamp(1));
  stream_a_manager_->PropagateUpdatesToMirrors(Timestamp(2), &stream_a_shard);

  EXPECT_EQ(1, schedule_count_);
  EXPECT_TRUE(node_->TryToBeginScheduling());
  EXPECT_NE(nullptr, cc_);
  MP_EXPECT_OK(node_->ProcessNode(cc_));
  node_->EndScheduling();
  EXPECT_EQ(1, schedule_count_);

  stream_a_manager_->ResetShard(&stream_a_shard);
  stream_a_shard.Add(new int(2), Timestamp(4));
  stream_a_shard.Add(new int(3), Timestamp(8));
  stream_a_manager_->PropagateUpdatesToMirrors(Timestamp(9), &stream_a_shard);
  // The packet at Timestamp 8 is left in the input queue.

  EXPECT_EQ(2, schedule_count_);
  EXPECT_TRUE(node_->TryToBeginScheduling());
  // Expects that a CalculatorContext has been prepared.
  EXPECT_NE(nullptr, cc_);
  MP_EXPECT_OK(node_->ProcessNode(cc_));
  node_->EndScheduling();
  EXPECT_EQ(3, schedule_count_);
  EXPECT_TRUE(node_->TryToBeginScheduling());

  stream_a_manager_->ResetShard(&stream_a_shard);
  stream_a_shard.Add(new int(4), Timestamp(16));
  stream_a_manager_->PropagateUpdatesToMirrors(Timestamp(17), &stream_a_shard);
  // The packet at Timestamp 16 is left in the input queue.

  EXPECT_EQ(3, schedule_count_);
  // The max parallelism is already reached.
  EXPECT_FALSE(node_->TryToBeginScheduling());
  EXPECT_NE(nullptr, cc_);
  MP_EXPECT_OK(node_->ProcessNode(cc_));
  node_->EndScheduling();
  EXPECT_EQ(4, schedule_count_);
  EXPECT_TRUE(node_->TryToBeginScheduling());

  EXPECT_NE(nullptr, cc_);
  MP_EXPECT_OK(node_->ProcessNode(cc_));

  cc_ = nullptr;
  node_->EndScheduling();
  // Expects that no CalculatorContext is prepared by EndScheduling().
  EXPECT_EQ(nullptr, cc_);
  EXPECT_EQ(4, schedule_count_);

  EXPECT_TRUE(node_->Prepared());
  EXPECT_TRUE(node_->Opened());
  EXPECT_FALSE(node_->Closed());

  EXPECT_EQ(1, CountCalculator::num_constructed_);
  EXPECT_EQ(1, CountCalculator::num_fill_expectations_);
  EXPECT_EQ(1, CountCalculator::num_open_);
  EXPECT_EQ(4, CountCalculator::num_process_);
  EXPECT_EQ(0, CountCalculator::num_close_);
  EXPECT_EQ(0, CountCalculator::num_destroyed_);
}

TEST_F(CalculatorNodeTest, Close) {
  InitializeEnvironment(/*use_tags=*/false);
  MP_ASSERT_OK(PrepareNodeForRun());

  SimulateParentOpenNode();
  MP_EXPECT_OK(node_->OpenNode());

  OutputStreamShard stream_a_shard;
  stream_a_shard.SetSpec(stream_a_manager_->Spec());
  stream_a_shard.Add(new int(1), Timestamp(1));
  stream_a_manager_->PropagateUpdatesToMirrors(Timestamp(2), &stream_a_shard);
  EXPECT_TRUE(node_->TryToBeginScheduling());
  stream_a_manager_->Close();
  // The max parallelism is already reached.
  EXPECT_FALSE(node_->TryToBeginScheduling());
  MP_EXPECT_OK(node_->ProcessNode(cc_));
  node_->EndScheduling();

  EXPECT_TRUE(node_->TryToBeginScheduling());
  MP_EXPECT_OK(node_->ProcessNode(cc_));
  EXPECT_TRUE(node_->Closed());
  EXPECT_EQ(2, schedule_count_);

  node_->EndScheduling();

  EXPECT_TRUE(node_->Prepared());
  EXPECT_TRUE(node_->Opened());
  EXPECT_TRUE(node_->Closed());

  EXPECT_EQ(2, schedule_count_);

  EXPECT_EQ(1, CountCalculator::num_constructed_);
  EXPECT_EQ(1, CountCalculator::num_fill_expectations_);
  EXPECT_EQ(1, CountCalculator::num_open_);
  EXPECT_EQ(1, CountCalculator::num_process_);
  EXPECT_EQ(1, CountCalculator::num_close_);
  EXPECT_EQ(0, CountCalculator::num_destroyed_);
}

TEST_F(CalculatorNodeTest, CleanupAfterRun) {
  InitializeEnvironment(/*use_tags=*/false);
  MP_ASSERT_OK(PrepareNodeForRun());

  SimulateParentOpenNode();
  MP_EXPECT_OK(node_->OpenNode());
  OutputStreamShard stream_a_shard;
  stream_a_shard.SetSpec(stream_a_manager_->Spec());
  stream_a_shard.Add(new int(1), Timestamp(1));
  stream_a_manager_->PropagateUpdatesToMirrors(Timestamp(2), &stream_a_shard);
  EXPECT_TRUE(node_->TryToBeginScheduling());
  stream_a_manager_->Close();
  // The max parallelism is already reached.
  EXPECT_FALSE(node_->TryToBeginScheduling());
  MP_EXPECT_OK(node_->ProcessNode(cc_));
  node_->EndScheduling();
  // Call ProcessNode again for the node to see the end of the stream.
  EXPECT_TRUE(node_->TryToBeginScheduling());
  MP_EXPECT_OK(node_->ProcessNode(cc_));
  node_->EndScheduling();
  // The max parallelism is already reached.
  EXPECT_FALSE(node_->TryToBeginScheduling());
  node_->CleanupAfterRun(absl::OkStatus());

  EXPECT_FALSE(node_->Prepared());
  EXPECT_FALSE(node_->Opened());
  EXPECT_FALSE(node_->Closed());

  EXPECT_EQ(2, schedule_count_);

  EXPECT_EQ(1, CountCalculator::num_constructed_);
  EXPECT_EQ(1, CountCalculator::num_fill_expectations_);
  EXPECT_EQ(1, CountCalculator::num_open_);
  EXPECT_EQ(1, CountCalculator::num_process_);
  EXPECT_EQ(1, CountCalculator::num_close_);
  EXPECT_EQ(1, CountCalculator::num_destroyed_);
}

void CalculatorNodeTest::TestCleanupAfterRunTwice() {
  MP_ASSERT_OK(PrepareNodeForRun());

  SimulateParentOpenNode();
  MP_EXPECT_OK(node_->OpenNode());
  OutputStreamShard stream_a_shard;
  stream_a_shard.SetSpec(stream_a_manager_->Spec());
  stream_a_shard.Add(new int(1), Timestamp(1));
  stream_a_manager_->PropagateUpdatesToMirrors(Timestamp(2), &stream_a_shard);
  EXPECT_TRUE(node_->TryToBeginScheduling());
  stream_a_manager_->Close();
  // The max parallelism is already reached.
  EXPECT_FALSE(node_->TryToBeginScheduling());
  MP_EXPECT_OK(node_->ProcessNode(cc_));
  node_->EndScheduling();
  // We should get Timestamp::Done here.
  EXPECT_TRUE(node_->TryToBeginScheduling());
  MP_EXPECT_OK(node_->ProcessNode(cc_));
  node_->EndScheduling();
  node_->CleanupAfterRun(absl::OkStatus());

  stream_a_manager_->PrepareForRun(nullptr);

  MP_ASSERT_OK(PrepareNodeForRun());

  SimulateParentOpenNode();
  MP_EXPECT_OK(node_->OpenNode());
  stream_a_manager_->ResetShard(&stream_a_shard);
  stream_a_shard.Add(new int(2), Timestamp(4));
  stream_a_shard.Add(new int(3), Timestamp(8));
  stream_a_manager_->PropagateUpdatesToMirrors(Timestamp(9), &stream_a_shard);
  EXPECT_TRUE(node_->TryToBeginScheduling());
  stream_a_manager_->Close();
  EXPECT_FALSE(node_->TryToBeginScheduling());
  MP_EXPECT_OK(node_->ProcessNode(cc_));
  node_->EndScheduling();
  EXPECT_TRUE(node_->TryToBeginScheduling());
  MP_EXPECT_OK(node_->ProcessNode(cc_));
  node_->EndScheduling();
  // We should get Timestamp::Done here.
  EXPECT_TRUE(node_->TryToBeginScheduling());
  MP_EXPECT_OK(node_->ProcessNode(cc_));
  node_->EndScheduling();
  // The max parallelism is already reached.
  EXPECT_FALSE(node_->TryToBeginScheduling());
  node_->CleanupAfterRun(absl::OkStatus());

  EXPECT_FALSE(node_->Prepared());
  EXPECT_FALSE(node_->Opened());
  EXPECT_FALSE(node_->Closed());

  EXPECT_EQ(5, schedule_count_);

  EXPECT_EQ(2, CountCalculator::num_constructed_);
  EXPECT_EQ(1, CountCalculator::num_fill_expectations_);
  EXPECT_EQ(2, CountCalculator::num_open_);
  EXPECT_EQ(3, CountCalculator::num_process_);
  EXPECT_EQ(2, CountCalculator::num_close_);
  EXPECT_EQ(2, CountCalculator::num_destroyed_);
}

TEST_F(CalculatorNodeTest, CleanupAfterRunTwice) {
  InitializeEnvironment(/*use_tags=*/false);
  TestCleanupAfterRunTwice();
}

TEST_F(CalculatorNodeTest, CleanupAfterRunTwiceWithTags) {
  InitializeEnvironment(/*use_tags=*/true);
  TestCleanupAfterRunTwice();
}

}  // namespace
}  // namespace mediapipe
