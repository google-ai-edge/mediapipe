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

#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace {

class CustomBoundCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    cc->Outputs().Index(0).SetNextTimestampBound(cc->InputTimestamp() + 1);
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(CustomBoundCalculator);

// Shows that ImmediateInputStreamHandler allows bounds propagation.
TEST(CalculatorGraphBounds, ImmediateHandlerBounds) {
  // CustomBoundCalculator produces only timestamp bounds.
  // The first PassThroughCalculator propagates bounds using SetOffset(0).
  // The second PassthroughCalculator delivers an output packet whenever the
  // first PassThroughCalculator delivers a timestamp bound.
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: 'input'
        node {
          calculator: 'CustomBoundCalculator'
          input_stream: 'input'
          output_stream: 'bounds'
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'bounds'
          output_stream: 'bounds_2'
          input_stream_handler {
            input_stream_handler: "ImmediateInputStreamHandler"
          }
        }
        node {
          calculator: 'PassThroughCalculator'
          input_stream: 'bounds_2'
          input_stream: 'input'
          output_stream: 'bounds_output'
          output_stream: 'output'
        }
      )");
  CalculatorGraph graph;
  std::vector<Packet> output_packets;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.ObserveOutputStream("output", [&](const Packet& p) {
    output_packets.push_back(p);
    return ::mediapipe::OkStatus();
  }));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // Add four packets into the graph.
  for (int i = 0; i < 4; ++i) {
    Packet p = MakePacket<int>(33).At(Timestamp(i));
    MP_ASSERT_OK(graph.AddPacketToInputStream("input", p));
  }

  // Four packets arrive at the output only if timestamp bounds are propagated.
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(output_packets.size(), 4);

  // Eventually four packets arrive.
  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_EQ(output_packets.size(), 4);
}

}  // namespace
}  // namespace mediapipe
