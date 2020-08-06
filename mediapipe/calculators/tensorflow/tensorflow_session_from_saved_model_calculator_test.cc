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

#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session_from_saved_model_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/tag_map_helper.h"
#include "mediapipe/framework/tool/validate_type.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

namespace mediapipe {

namespace {

namespace tf = ::tensorflow;

std::string GetSavedModelDir() {
  std::string out_path =
      file::JoinPath("./", "mediapipe/calculators/tensorflow/testdata/",
                     "tensorflow_saved_model/00000000");
  return out_path;
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

class TensorFlowSessionFromSavedModelCalculatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    extendable_options_.Clear();
    options_ = extendable_options_.MutableExtension(
        TensorFlowSessionFromSavedModelCalculatorOptions::ext);
    options_->set_saved_model_path(GetSavedModelDir());
  }

  CalculatorOptions extendable_options_;
  TensorFlowSessionFromSavedModelCalculatorOptions* options_;
};

TEST_F(TensorFlowSessionFromSavedModelCalculatorTest,
       CreatesPacketWithGraphAndBindings) {
  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromSavedModelCalculator"
        output_side_packet: "SESSION:tf_model"
        options {
          [mediapipe.TensorFlowSessionFromSavedModelCalculatorOptions.ext]: {
            $0
          }
        })",
                                           options_->DebugString()));
  MP_ASSERT_OK(runner.Run());
  const TensorFlowSession& session =
      runner.OutputSidePackets().Tag("SESSION").Get<TensorFlowSession>();
  // Session must be set.
  ASSERT_NE(session.session, nullptr);

  // Bindings are inserted.
  EXPECT_EQ(session.tag_to_tensor_map.size(), 4);

  // For some reason, EXPECT_EQ and EXPECT_NE are not working with iterators.
  EXPECT_FALSE(session.tag_to_tensor_map.find("A") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("B") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("MULTIPLIED") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("EXPENSIVE") ==
               session.tag_to_tensor_map.end());
  // Sanity: find() actually returns a reference to end() if element not
  // found.
  EXPECT_TRUE(session.tag_to_tensor_map.find("Z") ==
              session.tag_to_tensor_map.end());

  EXPECT_EQ(session.tag_to_tensor_map.at("A"), "a:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("B"), "b:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("MULTIPLIED"), "multiplied:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("EXPENSIVE"), "expensive:0");
}

TEST_F(TensorFlowSessionFromSavedModelCalculatorTest,
       CreateSessionFromSidePacket) {
  options_->clear_saved_model_path();
  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromSavedModelCalculator"
        input_side_packet: "STRING_SAVED_MODEL_PATH:saved_model_dir"
        output_side_packet: "SESSION:tf_model"
        options {
          [mediapipe.TensorFlowSessionFromSavedModelCalculatorOptions.ext]: {
            $0
          }
        })",
                                           options_->DebugString()));
  runner.MutableSidePackets()->Tag("STRING_SAVED_MODEL_PATH") =
      MakePacket<std::string>(GetSavedModelDir());
  MP_ASSERT_OK(runner.Run());
  const TensorFlowSession& session =
      runner.OutputSidePackets().Tag("SESSION").Get<TensorFlowSession>();
  // Session must be set.
  ASSERT_NE(session.session, nullptr);
}

// Integration test. Verifies that TensorFlowInferenceCalculator correctly
// consumes the Packet emitted by this factory.
TEST_F(TensorFlowSessionFromSavedModelCalculatorTest,
       ProducesPacketUsableByTensorFlowInferenceCalculator) {
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          absl::Substitute(R"(
      node {
        calculator: "TensorFlowInferenceCalculator"
        input_side_packet: "SESSION:tf_model"
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
        calculator: "TensorFlowSessionFromSavedModelCalculator"
        output_side_packet: "SESSION:tf_model"
        options {
          [mediapipe.TensorFlowSessionFromSavedModelCalculatorOptions.ext]: {
            $0
          }
        }
      }
      input_stream: "a_tensor"
  )",
                           options_->DebugString()));

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
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

TEST_F(TensorFlowSessionFromSavedModelCalculatorTest,
       GetsBundleGivenParentDirectory) {
  options_->set_saved_model_path(
      std::string(file::SplitPath(GetSavedModelDir()).first));
  options_->set_load_latest_model(true);

  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromSavedModelCalculator"
        output_side_packet: "SESSION:tf_model"
        options {
          [mediapipe.TensorFlowSessionFromSavedModelCalculatorOptions.ext]: {
            $0
          }
        })",
                                           options_->DebugString()));
  MP_ASSERT_OK(runner.Run());
  const TensorFlowSession& session =
      runner.OutputSidePackets().Tag("SESSION").Get<TensorFlowSession>();
  // Session must be set.
  ASSERT_NE(session.session, nullptr);
}

TEST_F(TensorFlowSessionFromSavedModelCalculatorTest,
       ConfiguresSessionGivenConfig) {
  options_->set_saved_model_path(
      std::string(file::SplitPath(GetSavedModelDir()).first));
  options_->set_load_latest_model(true);
  options_->mutable_session_config()->mutable_device_count()->insert(
      {"CPU", 10});
  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromSavedModelCalculator"
        output_side_packet: "SESSION:tf_model"
        options {
          [mediapipe.TensorFlowSessionFromSavedModelCalculatorOptions.ext]: {
            $0
          }
        })",
                                           options_->DebugString()));
  MP_ASSERT_OK(runner.Run());
  const TensorFlowSession& session =
      runner.OutputSidePackets().Tag("SESSION").Get<TensorFlowSession>();
  // Session must be set.
  ASSERT_NE(session.session, nullptr);
  std::vector<tensorflow::DeviceAttributes> devices;
  ASSERT_EQ(session.session->ListDevices(&devices), tensorflow::Status::OK());
  EXPECT_THAT(devices.size(), 10);
}

}  // namespace
}  // namespace mediapipe
