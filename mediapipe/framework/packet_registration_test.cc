// Copyright 2020 The MediaPipe Authors.
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

#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_test.pb.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Stream;

namespace test_ns {

constexpr char kOutTag[] = "OUT";
constexpr char kInTag[] = "IN";

class TestSinkCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kInTag).Set<mediapipe::InputOnlyProto>();
    cc->Outputs().Tag(kOutTag).Set<int>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    int x = cc->Inputs().Tag(kInTag).Get<mediapipe::InputOnlyProto>().x();
    cc->Outputs().Tag(kOutTag).AddPacket(
        MakePacket<int>(x).At(cc->InputTimestamp()));
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(TestSinkCalculator);

}  // namespace test_ns

TEST(PacketRegistrationTest, InputTypeRegistration) {
  using testing::Contains;
  ASSERT_EQ(mediapipe::InputOnlyProto{}.GetTypeName(),
            "mediapipe.InputOnlyProto");
  EXPECT_THAT(packet_internal::MessageHolderRegistry::GetRegisteredNames(),
              Contains("mediapipe.InputOnlyProto"));
}

TEST(PacketRegistrationTest, AdoptingRegisteredProtoWorks) {
  CalculatorGraphConfig config;
  {
    Graph graph;
    Stream<mediapipe::InputOnlyProto> input =
        graph.In(0).SetName("in").Cast<mediapipe::InputOnlyProto>();

    auto& sink_node = graph.AddNode("TestSinkCalculator");
    input.ConnectTo(sink_node.In(test_ns::kInTag));
    Stream<int> output = sink_node.Out(test_ns::kOutTag).Cast<int>();

    output.ConnectTo(graph.Out(0)).SetName("out");

    config = graph.GetConfig();
  }

  CalculatorGraph calculator_graph;
  MP_ASSERT_OK(calculator_graph.Initialize(std::move(config)));
  MP_ASSERT_OK(calculator_graph.StartRun({}));

  int value = 10;
  auto proto = std::make_unique<mediapipe::InputOnlyProto>();
  proto->set_x(value);
  MP_ASSERT_OK(calculator_graph.AddPacketToInputStream(
      "in", Adopt(proto.release()).At(Timestamp(0))));
  MP_ASSERT_OK(calculator_graph.WaitUntilIdle());
}

}  // namespace
}  // namespace mediapipe
