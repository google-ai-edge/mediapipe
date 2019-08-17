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

// A Calculator that selects an input stream from "INPUT:0", "INPUT:1", ...,
// using the integer value (0, 1, ...) in the packet on the "SELECT" input
// stream, and passes the packet on the selected input stream to the "OUTPUT"
// output stream.
//
// Note that this calculator defaults to use MuxInputStreamHandler, which is
// required for this calculator.
class MuxCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("SELECT").Set<int>();
    CollectionItemId data_input_id = cc->Inputs().BeginId("INPUT");
    PacketType* data_input0 = &cc->Inputs().Get(data_input_id);
    data_input0->SetAny();
    ++data_input_id;
    for (; data_input_id < cc->Inputs().EndId("INPUT"); ++data_input_id) {
      cc->Inputs().Get(data_input_id).SetSameAs(data_input0);
    }
    RET_CHECK_EQ(cc->Outputs().NumEntries(), 1);
    cc->Outputs().Tag("OUTPUT").SetSameAs(data_input0);

    // Assign this calculator's default InputStreamHandler.
    cc->SetInputStreamHandler("MuxInputStreamHandler");
    MediaPipeOptions options;
    cc->SetInputStreamHandlerOptions(options);

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    select_input_ = cc->Inputs().GetId("SELECT", 0);
    data_input_base_ = cc->Inputs().GetId("INPUT", 0);
    num_data_inputs_ = cc->Inputs().NumEntries("INPUT");
    output_ = cc->Outputs().GetId("OUTPUT", 0);
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    int select = cc->Inputs().Get(select_input_).Get<int>();
    RET_CHECK(0 <= select && select < num_data_inputs_);
    if (!cc->Inputs().Get(data_input_base_ + select).IsEmpty()) {
      cc->Outputs().Get(output_).AddPacket(
          cc->Inputs().Get(data_input_base_ + select).Value());
    }
    return ::mediapipe::OkStatus();
  }

 private:
  CollectionItemId select_input_;
  CollectionItemId data_input_base_;
  int num_data_inputs_ = 0;
  CollectionItemId output_;
};

REGISTER_CALCULATOR(MuxCalculator);

}  // namespace mediapipe
