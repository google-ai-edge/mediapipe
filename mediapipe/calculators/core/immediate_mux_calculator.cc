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
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// This Calculator multiplexes several input streams into a single
// output stream, dropping input packets with timestamps older than the
// last output packet.  In case two packets arrive with the same timestamp,
// the packet with the lower stream index will be output and the rest will
// be dropped.
//
// This Calculator optionally produces a finish inidicator as its second
// output stream.  One indicator packet is produced for each input packet
// received.
//
// This Calculator can be used with an ImmediateInputStreamHandler or with the
// default ISH. Note that currently ImmediateInputStreamHandler seems to
// interfere with timestamp bound propagation, so it is better to use the
// default unless the immediate one is needed. (b/118387598)
//
// This Calculator is designed to work with a Demux calculator such as
// the RoundRobinDemuxCalculator.  Therefore, packets from different
// input streams are normally not expected to have the same timestamp.
//
// NOTE: this calculator can drop packets non-deterministically, depending on
// how fast the input streams are fed. In most cases, MuxCalculator should be
// preferred. In particular, dropping packets can interfere with rate limiting
// mechanisms.
class ImmediateMuxCalculator : public CalculatorBase {
 public:
  // This calculator combines any set of input streams into a single
  // output stream.  All input stream types must match the output stream type.
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  // Passes any input packet to the output stream immediately, unless the
  // packet timestamp is lower than a previously passed packet.
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Open(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(ImmediateMuxCalculator);

::mediapipe::Status ImmediateMuxCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Outputs().NumEntries() >= 1 && cc->Outputs().NumEntries() <= 2)
      << "This calculator produces only one or two output streams.";
  cc->Outputs().Index(0).SetAny();
  if (cc->Outputs().NumEntries() >= 2) {
    cc->Outputs().Index(1).Set<bool>();
  }
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    cc->Inputs().Index(i).SetSameAs(&cc->Outputs().Index(0));
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImmediateMuxCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImmediateMuxCalculator::Process(CalculatorContext* cc) {
  // Pass along the first packet, unless it has been superseded.
  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    const Packet& packet = cc->Inputs().Index(i).Value();
    if (!packet.IsEmpty()) {
      if (packet.Timestamp() >= cc->Outputs().Index(0).NextTimestampBound()) {
        cc->Outputs().Index(0).AddPacket(packet);
      } else {
        LOG_FIRST_N(WARNING, 5)
            << "Dropping a packet with timestamp " << packet.Timestamp();
      }
      if (cc->Outputs().NumEntries() >= 2) {
        Timestamp output_timestamp = std::max(
            cc->InputTimestamp(), cc->Outputs().Index(1).NextTimestampBound());
        cc->Outputs().Index(1).Add(new bool(true), output_timestamp);
      }
    }
  }
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
