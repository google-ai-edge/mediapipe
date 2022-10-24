// Copyright 2022 The MediaPipe Authors.
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

#include <cstring>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/options_map.h"

namespace mediapipe {
namespace {

using ::testing::StrEq;

absl::StatusOr<std::string> RunTextToTensorCalculator(absl::string_view text) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        input_stream: "text"
        output_stream: "tensors"
        node {
          calculator: "TextToTensorCalculator"
          input_stream: "TEXT:text"
          output_stream: "TENSORS:tensors"
        }
      )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensors", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(graph_config));
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "text", MakePacket<std::string>(text).At(Timestamp(0))));
  MP_RETURN_IF_ERROR(graph.WaitUntilIdle());

  if (output_packets.size() != 1) {
    return absl::InvalidArgumentError(absl::Substitute(
        "output_packets has size $0, expected 1", output_packets.size()));
  }
  const std::vector<Tensor>& tensor_vec =
      output_packets[0].Get<std::vector<Tensor>>();
  if (tensor_vec.size() != 1) {
    return absl::InvalidArgumentError(absl::Substitute(
        "tensor_vec has size $0, expected 1", tensor_vec.size()));
  }
  if (tensor_vec[0].element_type() != Tensor::ElementType::kChar) {
    return absl::InvalidArgumentError("Expected tensor element type kChar");
  }
  const char* buffer = tensor_vec[0].GetCpuReadView().buffer<char>();
  return std::string(buffer, text.length());
}

TEST(TextToTensorCalculatorTest, FooBarBaz) {
  EXPECT_THAT(RunTextToTensorCalculator("Foo. Bar? Baz!"),
              IsOkAndHolds(StrEq("Foo. Bar? Baz!")));
}

TEST(TextToTensorCalculatorTest, Empty) {
  EXPECT_THAT(RunTextToTensorCalculator(""), IsOkAndHolds(StrEq("")));
}

}  // namespace
}  // namespace mediapipe
