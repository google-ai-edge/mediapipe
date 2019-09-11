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

#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/port/threadpool.h"
#include "mediapipe/framework/stream_handler/fixed_size_input_stream_handler.pb.h"

namespace mediapipe {

namespace {

// Copied from mux_input_stream_handler_test.cc, and removed ISH from the graph
//
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
          # MuxInputStreamHandler set in GetContract().
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

// Copied from pass_through_calculator.cc, and modified to specify
// InputStreamHandler.
//
// A Calculator that simply passes its input Packets and header through,
// unchanged.  The inputs may be specified by tag or index.  The outputs
// must match the inputs exactly.  Any number of input side packets may
// also be specified.  If output side packets are specified, they must
// match the input side packets exactly and the Calculator passes its
// input side packets through, unchanged.  Otherwise, the input side
// packets will be ignored (allowing PassThroughCalculator to be used to
// test internal behavior).  Any options may be specified and will be
// ignored.
class FixedPassThroughCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    if (!cc->Inputs().TagMap()->SameAs(*cc->Outputs().TagMap())) {
      return ::mediapipe::InvalidArgumentError(
          "Input and output streams to PassThroughCalculator must use "
          "matching tags and indexes.");
    }
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      cc->Inputs().Get(id).SetAny();
      cc->Outputs().Get(id).SetSameAs(&cc->Inputs().Get(id));
    }
    for (CollectionItemId id = cc->InputSidePackets().BeginId();
         id < cc->InputSidePackets().EndId(); ++id) {
      cc->InputSidePackets().Get(id).SetAny();
    }
    if (cc->OutputSidePackets().NumEntries() != 0) {
      if (!cc->InputSidePackets().TagMap()->SameAs(
              *cc->OutputSidePackets().TagMap())) {
        return ::mediapipe::InvalidArgumentError(
            "Input and output side packets to PassThroughCalculator must use "
            "matching tags and indexes.");
      }
      for (CollectionItemId id = cc->InputSidePackets().BeginId();
           id < cc->InputSidePackets().EndId(); ++id) {
        cc->OutputSidePackets().Get(id).SetSameAs(
            &cc->InputSidePackets().Get(id));
      }
    }

    // Assign this calculator's InputStreamHandler and options.
    cc->SetInputStreamHandler("FixedSizeInputStreamHandler");
    MediaPipeOptions options;
    options.MutableExtension(FixedSizeInputStreamHandlerOptions::ext)
        ->set_fixed_min_size(2);
    options.MutableExtension(FixedSizeInputStreamHandlerOptions::ext)
        ->set_trigger_queue_size(2);
    options.MutableExtension(FixedSizeInputStreamHandlerOptions::ext)
        ->set_target_queue_size(2);
    cc->SetInputStreamHandlerOptions(options);

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      if (!cc->Inputs().Get(id).Header().IsEmpty()) {
        cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
      }
    }
    if (cc->OutputSidePackets().NumEntries() != 0) {
      for (CollectionItemId id = cc->InputSidePackets().BeginId();
           id < cc->InputSidePackets().EndId(); ++id) {
        cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
      }
    }
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    cc->GetCounter("PassThrough")->Increment();
    if (cc->Inputs().NumEntries() == 0) {
      return tool::StatusStop();
    }
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      if (!cc->Inputs().Get(id).IsEmpty()) {
        VLOG(3) << "Passing " << cc->Inputs().Get(id).Name() << " to "
                << cc->Outputs().Get(id).Name() << " at "
                << cc->InputTimestamp().DebugString();
        cc->Outputs().Get(id).AddPacket(cc->Inputs().Get(id).Value());
      }
    }
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(FixedPassThroughCalculator);

// Copied from fixed_size_input_stream_handler_test.cc, and modified graph.
//
// Tests FixedSizeInputStreamHandler with several input streams running
// asynchronously in parallel.
TEST(FixedSizeInputStreamHandlerTest, ParallelWriteAndRead) {
#define NUM_STREAMS 4
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"(
            input_stream: "in_0"
            input_stream: "in_1"
            input_stream: "in_2"
            input_stream: "in_3"
            node {
              calculator: "FixedPassThroughCalculator"
              input_stream: "in_0"
              input_stream: "in_1"
              input_stream: "in_2"
              input_stream: "in_3"
              output_stream: "out_0"
              output_stream: "out_1"
              output_stream: "out_2"
              output_stream: "out_3"
              # FixedSizeInputStreamHandler set in GetContract()
            })");
  std::vector<Packet> output_packets[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; ++i) {
    tool::AddVectorSink(absl::StrCat("out_", i), &graph_config,
                        &output_packets[i]);
  }
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));

  {
    ::mediapipe::ThreadPool pool(NUM_STREAMS);
    pool.StartWorkers();

    // Start writers.
    for (int w = 0; w < NUM_STREAMS; ++w) {
      pool.Schedule([&, w]() {
        std::string stream_name = absl::StrCat("in_", w);
        for (int i = 0; i < 50; ++i) {
          Packet p = MakePacket<int>(i).At(Timestamp(i));
          MP_EXPECT_OK(graph.AddPacketToInputStream(stream_name, p));
          absl::SleepFor(absl::Microseconds(100));
        }
      });
    }
  }

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  for (int i = 0; i < NUM_STREAMS; ++i) {
    EXPECT_EQ(output_packets[i].size(), output_packets[0].size());
    for (int j = 0; j < output_packets[i].size(); j++) {
      EXPECT_EQ(output_packets[i][j].Get<int>(),
                output_packets[0][j].Get<int>());
    }
  }
}

}  // namespace
}  // namespace mediapipe
