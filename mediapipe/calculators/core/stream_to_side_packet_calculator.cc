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
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace api2 {

// A calculator that takes a packet of an input stream and converts it to an
// output side packet. This calculator only works under the assumption that the
// input stream only has a single packet passing through.
//
// Example config:
// node {
//   calculator: "StreamToSidePacketCalculator"
//   input_stream: "stream"
//   output_side_packet: "side_packet"
// }
class StreamToSidePacketCalculator : public Node {
 public:
  static constexpr Input<AnyType>::Optional kIn{""};
  static constexpr SideOutput<SameType<kIn>> kOut{""};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Process(mediapipe::CalculatorContext* cc) override {
    kOut(cc).Set(
        kIn(cc).packet().As<AnyType>().At(mediapipe::Timestamp::Unset()));
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(StreamToSidePacketCalculator);

}  // namespace api2
}  // namespace mediapipe
