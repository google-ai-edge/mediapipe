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

#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session_from_frozen_graph_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/tag_map_helper.h"
#include "mediapipe/framework/tool/validate_type.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace mediapipe {

namespace {

namespace tf = ::tensorflow;

std::string GetGraphDefPath() {
  return mediapipe::file::JoinPath("./",
                                   "mediapipe/calculators/tensorflow/"
                                   "testdata/frozen_graph_def.pb");
}

// Helper function that creates Tensor INT32 matrix with size 1x3.
tf::Tensor TensorMatrix1x3(const int v1, const int v2, const int v3) {
  tf::Tensor tensor(tf::DT_INT32,
                    tf::TensorShape(std::vector<tf::int64>({1, 3})));
  auto matrix = tensor.matrix<int32>();
  matrix(0, 0) = v1;
  matrix(0, 1) = v2;
  matrix(0, 2) = v3;
  return tensor;
}

class TensorFlowSessionFromFrozenGraphCalculatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    extendable_options_.Clear();
    calculator_options_ = extendable_options_.MutableExtension(
        TensorFlowSessionFromFrozenGraphCalculatorOptions::ext);
    calculator_options_->set_graph_proto_path(GetGraphDefPath());
    (*calculator_options_->mutable_tag_to_tensor_names())["MULTIPLIED"] =
        "multiplied:0";
    (*calculator_options_->mutable_tag_to_tensor_names())["A"] = "a:0";
    (*calculator_options_->mutable_tag_to_tensor_names())["B"] = "b:0";
    calculator_options_->mutable_config()->set_intra_op_parallelism_threads(1);
    calculator_options_->mutable_config()->set_inter_op_parallelism_threads(2);
    calculator_options_->set_preferred_device_id("/device:CPU:0");
  }

  void VerifySignatureMap(const TensorFlowSession& session) {
    // Session must be set.
    ASSERT_NE(session.session, nullptr);

    // Bindings are inserted.
    EXPECT_EQ(session.tag_to_tensor_map.size(), 3);

    // For some reason, EXPECT_EQ and EXPECT_NE are not working with iterators.
    EXPECT_FALSE(session.tag_to_tensor_map.find("A") ==
                 session.tag_to_tensor_map.end());
    EXPECT_FALSE(session.tag_to_tensor_map.find("B") ==
                 session.tag_to_tensor_map.end());
    EXPECT_FALSE(session.tag_to_tensor_map.find("MULTIPLIED") ==
                 session.tag_to_tensor_map.end());
    // Sanity: find() actually returns a reference to end() if element not
    // found.
    EXPECT_TRUE(session.tag_to_tensor_map.find("Z") ==
                session.tag_to_tensor_map.end());

    EXPECT_EQ(session.tag_to_tensor_map.at("A"), "a:0");
    EXPECT_EQ(session.tag_to_tensor_map.at("B"), "b:0");
    EXPECT_EQ(session.tag_to_tensor_map.at("MULTIPLIED"), "multiplied:0");
  }

  CalculatorOptions extendable_options_;
  TensorFlowSessionFromFrozenGraphCalculatorOptions* calculator_options_;
};

TEST_F(TensorFlowSessionFromFrozenGraphCalculatorTest,
       CreatesPacketWithGraphAndBindings) {
  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromFrozenGraphCalculator"
        output_side_packet: "SESSION:tf_model"
        options {
          [mediapipe.TensorFlowSessionFromFrozenGraphCalculatorOptions.ext]: {
            $0
          }
        })",
                                           calculator_options_->DebugString()));

  MP_ASSERT_OK(runner.Run());
  const TensorFlowSession& session =
      runner.OutputSidePackets().Tag("SESSION").Get<TensorFlowSession>();
  VerifySignatureMap(session);
}

// Integration test. Verifies that TensorFlowInferenceCalculator correctly
// consumes the Packet emitted by this calculator.
TEST_F(TensorFlowSessionFromFrozenGraphCalculatorTest,
       ProducesPacketUsableByTensorFlowInferenceCalculator) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          absl::Substitute(R"(
      node {
        calculator: "TensorFlowInferenceCalculator"
        input_side_packet: "SESSION:session"
        input_stream: "A:a_tensor"
        output_stream: "MULTIPLIED:multiplied_tensor"
        options {
          [mediapipe.TensorFlowInferenceCalculatorOptions.ext] {
            batch_size: 5
            add_batch_dim_to_tensors: false
          }
        }
      }

      node {
        calculator: "TensorFlowSessionFromFrozenGraphCalculator"
        output_side_packet: "SESSION:session"
        options {
          [mediapipe.TensorFlowSessionFromFrozenGraphCalculatorOptions.ext]: {
            $0
          }
        }
      }
      input_stream: "a_tensor"
  )",
                           calculator_options_->DebugString()));

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  StatusOrPoller status_or_poller =
      graph.AddOutputStreamPoller("multiplied_tensor");
  ASSERT_TRUE(status_or_poller.ok());
  OutputStreamPoller poller = std::move(status_or_poller.ValueOrDie());

  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "a_tensor",
      Adopt(new auto(TensorMatrix1x3(1, -1, 10))).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseInputStream("a_tensor"));

  Packet packet;
  ASSERT_TRUE(poller.Next(&packet));
  // input tensor gets multiplied by [[3, 2, 1]]. Expected output:
  tf::Tensor expected_multiplication = TensorMatrix1x3(3, -2, 10);
  EXPECT_EQ(expected_multiplication.DebugString(),
            packet.Get<tf::Tensor>().DebugString());

  ASSERT_FALSE(poller.Next(&packet));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(TensorFlowSessionFromFrozenGraphCalculatorTest,
       CreatesPacketWithGraphAndBindingsFromInputSidePacket) {
  calculator_options_->clear_graph_proto_path();
  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromFrozenGraphCalculator"
        input_side_packet: "STRING_MODEL:model"
        output_side_packet: "SESSION:session"
        options {
          [mediapipe.TensorFlowSessionFromFrozenGraphCalculatorOptions.ext]: {
            $0
          }
        })",
                                           calculator_options_->DebugString()));

  std::string serialized_graph_contents;
  MP_EXPECT_OK(mediapipe::file::GetContents(GetGraphDefPath(),
                                            &serialized_graph_contents));
  runner.MutableSidePackets()->Tag("STRING_MODEL") =
      Adopt(new std::string(serialized_graph_contents));
  MP_ASSERT_OK(runner.Run());

  const TensorFlowSession& session =
      runner.OutputSidePackets().Tag("SESSION").Get<TensorFlowSession>();
  VerifySignatureMap(session);
}

TEST_F(
    TensorFlowSessionFromFrozenGraphCalculatorTest,
    CreatesPacketWithGraphAndBindingsFromInputSidePacketStringModelFilePath) {
  calculator_options_->clear_graph_proto_path();
  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromFrozenGraphCalculator"
        input_side_packet: "STRING_MODEL_FILE_PATH:file_path"
        output_side_packet: "SESSION:session"
        options {
          [mediapipe.TensorFlowSessionFromFrozenGraphCalculatorOptions.ext]: {
            $0
          }
        })",
                                           calculator_options_->DebugString()));
  runner.MutableSidePackets()->Tag("STRING_MODEL_FILE_PATH") =
      Adopt(new std::string(GetGraphDefPath()));
  MP_ASSERT_OK(runner.Run());

  const TensorFlowSession& session =
      runner.OutputSidePackets().Tag("SESSION").Get<TensorFlowSession>();
  VerifySignatureMap(session);
}

TEST_F(TensorFlowSessionFromFrozenGraphCalculatorTest,
       CheckFailureForOptionsAndInputsProvideGraphDefProto) {
  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromFrozenGraphCalculator"
        input_side_packet: "STRING_MODEL_FILE_PATH:file_path"
        output_side_packet: "SESSION:session"
        options {
          [mediapipe.TensorFlowSessionFromFrozenGraphCalculatorOptions.ext]: {
            $0
          }
        })",
                                           calculator_options_->DebugString()));
  runner.MutableSidePackets()->Tag("STRING_MODEL_FILE_PATH") =
      Adopt(new std::string(GetGraphDefPath()));
  auto run_status = runner.Run();
  EXPECT_THAT(
      run_status.message(),
      ::testing::HasSubstr("Must have exactly one of graph_proto_path"));
}

TEST_F(TensorFlowSessionFromFrozenGraphCalculatorTest,
       CheckFailureForAllInputsProvideGraphDefProto) {
  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromFrozenGraphCalculator"
        input_side_packet: "STRING_MODEL_FILE_PATH:file_path"
        input_side_packet: "STRING_MODEL:model"
        output_side_packet: "SESSION:session"
        options {
          [mediapipe.TensorFlowSessionFromFrozenGraphCalculatorOptions.ext]: {
            $0
          }
        })",
                                           calculator_options_->DebugString()));
  runner.MutableSidePackets()->Tag("STRING_MODEL_FILE_PATH") =
      Adopt(new std::string(GetGraphDefPath()));
  std::string serialized_graph_contents;
  MP_EXPECT_OK(mediapipe::file::GetContents(GetGraphDefPath(),
                                            &serialized_graph_contents));
  runner.MutableSidePackets()->Tag("STRING_MODEL") =
      Adopt(new std::string(serialized_graph_contents));
  auto run_status = runner.Run();
  EXPECT_THAT(
      run_status.message(),
      ::testing::HasSubstr("Must have exactly one of graph_proto_path"));
}

TEST_F(TensorFlowSessionFromFrozenGraphCalculatorTest,
       CheckFailureForOnlyBothInputSidePacketsProvideGraphDefProto) {
  calculator_options_->clear_graph_proto_path();
  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromFrozenGraphCalculator"
        input_side_packet: "STRING_MODEL_FILE_PATH:file_path"
        input_side_packet: "STRING_MODEL:model"
        output_side_packet: "SESSION:session"
        options {
          [mediapipe.TensorFlowSessionFromFrozenGraphCalculatorOptions.ext]: {
            $0
          }
        })",
                                           calculator_options_->DebugString()));
  runner.MutableSidePackets()->Tag("STRING_MODEL_FILE_PATH") =
      Adopt(new std::string(GetGraphDefPath()));
  std::string serialized_graph_contents;
  MP_EXPECT_OK(mediapipe::file::GetContents(GetGraphDefPath(),
                                            &serialized_graph_contents));
  runner.MutableSidePackets()->Tag("STRING_MODEL") =
      Adopt(new std::string(serialized_graph_contents));
  auto run_status = runner.Run();
  EXPECT_THAT(
      run_status.message(),
      ::testing::HasSubstr("Must have exactly one of graph_proto_path"));
}

TEST_F(TensorFlowSessionFromFrozenGraphCalculatorTest,
       CheckInitializationOpName) {
  calculator_options_->add_initialization_op_names("multiplied:0");
  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromFrozenGraphCalculator"
        output_side_packet: "SESSION:session"
        options {
          [mediapipe.TensorFlowSessionFromFrozenGraphCalculatorOptions.ext]: {
            $0
          }
        })",
                                           calculator_options_->DebugString()));
  MP_ASSERT_OK(runner.Run());

  const TensorFlowSession& session =
      runner.OutputSidePackets().Tag("SESSION").Get<TensorFlowSession>();
  VerifySignatureMap(session);
}

}  // namespace
}  // namespace mediapipe
