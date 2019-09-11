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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

// A regression test for b/31620439. MuxInputStreamHandler's accesses to the
// control and data streams should be atomic so that it has a consistent view
// of the two streams. None of the CHECKs in the GetNodeReadiness() method of
// MuxInputStreamHandler should fail when running this test.
TEST(MuxInputStreamHandlerTest, AtomicAccessToControlAndDataStreams) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "input"
        node {
          calculator: "RoundRobinDemuxCalculator"
          input_stream: "input"
          output_stream: "OUTPUT:0:input0"
          output_stream: "OUTPUT:1:input1"
          output_stream: "OUTPUT:2:input2"
          output_stream: "OUTPUT:3:input3"
          output_stream: "OUTPUT:4:input4"
          output_stream: "SELECT:select"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input0"
          output_stream: "output0"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input1"
          output_stream: "output1"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input2"
          output_stream: "output2"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input3"
          output_stream: "output3"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "input4"
          output_stream: "output4"
        }
        node {
          calculator: "MuxCalculator"
          input_stream: "INPUT:0:output0"
          input_stream: "INPUT:1:output1"
          input_stream: "INPUT:2:output2"
          input_stream: "INPUT:3:output3"
          input_stream: "INPUT:4:output4"
          input_stream: "SELECT:select"
          output_stream: "OUTPUT:output"
          input_stream_handler { input_stream_handler: "MuxInputStreamHandler" }
        })");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  for (int i = 0; i < 2000; ++i) {
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input", Adopt(new int(i)).At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace
}  // namespace mediapipe
