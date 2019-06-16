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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

// Forwards the input packet to one of the n output streams "OUTPUT:0",
// "OUTPUT:1", ..., in round robin fashion.  The index of the selected output
// stream is emitted to the output stream "SELECT".  If not needed, the
// output stream "SELECT" may be omitted.
//
// Designed to run graph bottlenecks in parallel and thus reduce graph
// processing latency by parallelizing.
//
// A simple example config is:
//
// node {
//   calculator: "RoundRobinDemuxCalculator"
//   input_stream: "signal"
//   output_stream: "OUTPUT:0:signal0"
//   output_stream: "OUTPUT:1:signal1"
//   output_stream: "SELECT:select"
// }
//
// node {
//   calculator: "SlowCalculator"
//   input_stream: "signal0"
//   output_stream: "output0"
// }
//
// node {
//   calculator: "SlowCalculator"
//   input_stream: "signal1"
//   output_stream: "output1"
// }
//
// node {
//   calculator: "MuxCalculator"
//   input_stream: "INPUT:0:output0"
//   input_stream: "INPUT:1:output1"
//   input_stream: "SELECT:select"
//   output_stream: "OUTPUT:output"
//   input_stream_handler {
//     input_stream_handler: "MuxInputStreamHandler"
//   }
// }
//
// which is essentially running the following configuration in parallel with a
// concurrency level of two:
//
// node {
//   calculator: "SlowCalculator"
//   input_stream: "signal"
//   output_stream: "output"
// }
//
// If SlowCalculator has more than one output stream, the user can group the
// output with MakePairCalculator, MakeVectorCalculator, or a similar variant to
// use it with MuxCalculator and later unpack, or can create new variants of
// MuxCalculator/MuxInputStreamHandler.
class RoundRobinDemuxCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK_EQ(cc->Inputs().NumEntries(), 1);
    cc->Inputs().Index(0).SetAny();
    if (cc->Outputs().HasTag("SELECT")) {
      cc->Outputs().Tag("SELECT").Set<int>();
    }
    for (CollectionItemId id = cc->Outputs().BeginId("OUTPUT");
         id < cc->Outputs().EndId("OUTPUT"); ++id) {
      cc->Outputs().Get(id).SetSameAs(&cc->Inputs().Index(0));
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    select_output_ = cc->Outputs().GetId("SELECT", 0);
    output_data_stream_index_ = 0;
    output_data_stream_base_ = cc->Outputs().GetId("OUTPUT", 0);
    num_output_data_streams_ = cc->Outputs().NumEntries("OUTPUT");
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    cc->Outputs()
        .Get(output_data_stream_base_ + output_data_stream_index_)
        .AddPacket(cc->Inputs().Index(0).Value());
    if (select_output_.IsValid()) {
      cc->Outputs()
          .Get(select_output_)
          .Add(new int(output_data_stream_index_), cc->InputTimestamp());
    }
    output_data_stream_index_ =
        (output_data_stream_index_ + 1) % num_output_data_streams_;
    return ::mediapipe::OkStatus();
  }

 private:
  CollectionItemId select_output_;
  CollectionItemId output_data_stream_base_;
  int num_output_data_streams_;
  int output_data_stream_index_;
};

REGISTER_CALCULATOR(RoundRobinDemuxCalculator);

}  // namespace mediapipe
