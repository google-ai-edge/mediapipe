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

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

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
class RoundRobinDemuxCalculator : public Node {
 public:
  static constexpr Input<AnyType> kIn{""};
  static constexpr Output<int>::Optional kSelect{"SELECT"};
  static constexpr Output<SameType<kIn>>::Multiple kOut{"OUTPUT"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kSelect, kOut);

  absl::Status Open(CalculatorContext* cc) override {
    output_data_stream_index_ = 0;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    kOut(cc)[output_data_stream_index_].Send(kIn(cc).packet());
    if (kSelect(cc).IsConnected()) {
      kSelect(cc).Send(output_data_stream_index_);
    }
    output_data_stream_index_ =
        (output_data_stream_index_ + 1) % kOut(cc).Count();
    return absl::OkStatus();
  }

 private:
  int output_data_stream_index_;
};

MEDIAPIPE_REGISTER_NODE(RoundRobinDemuxCalculator);

}  // namespace api2
}  // namespace mediapipe
