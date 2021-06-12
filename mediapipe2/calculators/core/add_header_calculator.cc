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
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {
namespace api2 {

// Attach the header from a stream or side input to another stream.
//
// The header stream (tag HEADER) must not have any packets in it.
//
// Before using this calculator, please think about changing your
// calculator to not need a header or to accept a separate stream with
// a header, that would be more future proof.
//
// Example usage 1:
// node {
//   calculator: "AddHeaderCalculator"
//   input_stream: "DATA:audio"
//   input_stream: "HEADER:audio_header"
//   output_stream: "audio_with_header"
// }
//
// Example usage 2:
// node {
//   calculator: "AddHeaderCalculator"
//   input_stream: "DATA:audio"
//   input_side_packet: "HEADER:audio_header"
//   output_stream: "audio_with_header"
// }
//
class AddHeaderCalculator : public Node {
 public:
  static constexpr Input<NoneType>::Optional kHeader{"HEADER"};
  static constexpr SideInput<AnyType>::Optional kHeaderSide{"HEADER"};
  static constexpr Input<AnyType> kData{"DATA"};
  static constexpr Output<SameType<kData>> kOut{""};

  MEDIAPIPE_NODE_CONTRACT(kHeader, kHeaderSide, kData, kOut);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    if (kHeader(cc).IsConnected() == kHeaderSide(cc).IsConnected()) {
      return absl::InvalidArgumentError(
          "Header must be provided via exactly one of side input and input "
          "stream");
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const PacketBase& header =
        kHeader(cc).IsConnected() ? kHeader(cc).Header() : kHeaderSide(cc);
    if (!header.IsEmpty()) {
      kOut(cc).SetHeader(header);
    }
    cc->SetOffset(0);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    kOut(cc).Send(kData(cc).packet());
    return absl::OkStatus();
  }
};

MEDIAPIPE_REGISTER_NODE(AddHeaderCalculator);

}  // namespace api2
}  // namespace mediapipe
