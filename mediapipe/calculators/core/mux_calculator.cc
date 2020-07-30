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

namespace {
constexpr char kSelectTag[] = "SELECT";
constexpr char kInputTag[] = "INPUT";
}  // namespace

// A Calculator that selects an input stream from "INPUT:0", "INPUT:1", ...,
// using the integer value (0, 1, ...) in the packet on the kSelectTag input
// stream, and passes the packet on the selected input stream to the "OUTPUT"
// output stream.
// The kSelectTag input can also be passed in as an input side packet, instead
// of as an input stream. Either of input stream or input side packet must be
// specified but not both.
//
// Note that this calculator defaults to use MuxInputStreamHandler, which is
// required for this calculator. However, it can be overridden to work with
// other InputStreamHandlers. Check out the unit tests on for an example usage
// with DefaultInputStreamHandler.
class MuxCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status CheckAndInitAllowDisallowInputs(
      CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kSelectTag) ^
              cc->InputSidePackets().HasTag(kSelectTag));
    if (cc->Inputs().HasTag(kSelectTag)) {
      cc->Inputs().Tag(kSelectTag).Set<int>();
    } else {
      cc->InputSidePackets().Tag(kSelectTag).Set<int>();
    }
    return ::mediapipe::OkStatus();
  }

  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK_OK(CheckAndInitAllowDisallowInputs(cc));
    CollectionItemId data_input_id = cc->Inputs().BeginId(kInputTag);
    PacketType* data_input0 = &cc->Inputs().Get(data_input_id);
    data_input0->SetAny();
    ++data_input_id;
    for (; data_input_id < cc->Inputs().EndId(kInputTag); ++data_input_id) {
      cc->Inputs().Get(data_input_id).SetSameAs(data_input0);
    }
    RET_CHECK_EQ(cc->Outputs().NumEntries(), 1);
    cc->Outputs().Tag("OUTPUT").SetSameAs(data_input0);

    cc->SetInputStreamHandler("MuxInputStreamHandler");
    MediaPipeOptions options;
    cc->SetInputStreamHandlerOptions(options);

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    use_side_packet_select_ = false;
    if (cc->InputSidePackets().HasTag(kSelectTag)) {
      use_side_packet_select_ = true;
      selected_index_ = cc->InputSidePackets().Tag(kSelectTag).Get<int>();
    } else {
      select_input_ = cc->Inputs().GetId(kSelectTag, 0);
    }
    data_input_base_ = cc->Inputs().GetId(kInputTag, 0);
    num_data_inputs_ = cc->Inputs().NumEntries(kInputTag);
    output_ = cc->Outputs().GetId("OUTPUT", 0);
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    int select = use_side_packet_select_
                     ? selected_index_
                     : cc->Inputs().Get(select_input_).Get<int>();
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
  bool use_side_packet_select_;
  int selected_index_;
};

REGISTER_CALCULATOR(MuxCalculator);

}  // namespace mediapipe
