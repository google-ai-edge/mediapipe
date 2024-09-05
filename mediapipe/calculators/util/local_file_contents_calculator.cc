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

#include <memory>
#include <string>

#include "mediapipe/calculators/util/local_file_contents_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe {

namespace {

constexpr char kFilePathTag[] = "FILE_PATH";
constexpr char kContentsTag[] = "CONTENTS";

}  // namespace

// The calculator takes the path to the local file as an input side packet and
// outputs the contents of that file.
//
// NOTE: file loading can be batched by providing multiple input/output side
// packets.
//
// Example config:
// node {
//   calculator: "LocalFileContentsCalculator"
//   input_side_packet: "FILE_PATH:file_path"
//   output_side_packet: "CONTENTS:contents"
// }
//
// node {
//   calculator: "LocalFileContentsCalculator"
//   input_side_packet: "FILE_PATH:0:file_path1"
//   input_side_packet: "FILE_PATH:1:file_path2"
//   ...
//   output_side_packet: "CONTENTS:0:contents1"
//   output_side_packet: "CONTENTS:1:contents2"
//   ...
// }
class LocalFileContentsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->InputSidePackets().HasTag(kFilePathTag))
        << "Missing PATH input side packet(s)";
    RET_CHECK(cc->OutputSidePackets().HasTag(kContentsTag))
        << "Missing CONTENTS output side packet(s)";

    RET_CHECK_EQ(cc->InputSidePackets().NumEntries(kFilePathTag),
                 cc->OutputSidePackets().NumEntries(kContentsTag))
        << "Same number of input streams and output streams is required.";

    for (CollectionItemId id = cc->InputSidePackets().BeginId(kFilePathTag);
         id != cc->InputSidePackets().EndId(kFilePathTag); ++id) {
      cc->InputSidePackets().Get(id).Set<std::string>();
    }

    for (CollectionItemId id = cc->OutputSidePackets().BeginId(kContentsTag);
         id != cc->OutputSidePackets().EndId(kContentsTag); ++id) {
      cc->OutputSidePackets().Get(id).Set<std::string>();
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    CollectionItemId input_id = cc->InputSidePackets().BeginId(kFilePathTag);
    CollectionItemId output_id = cc->OutputSidePackets().BeginId(kContentsTag);
    auto options = cc->Options<mediapipe::LocalFileContentsCalculatorOptions>();

    // Number of inputs and outpus is the same according to the contract.
    for (; input_id != cc->InputSidePackets().EndId(kFilePathTag);
         ++input_id, ++output_id) {
      std::string file_path =
          cc->InputSidePackets().Get(input_id).Get<std::string>();
      MP_ASSIGN_OR_RETURN(file_path, PathToResourceAsFile(file_path));

      std::string contents;
      MP_RETURN_IF_ERROR(cc->GetResources().ReadContents(
          file_path, contents, {/* read_as_binary= */ !options.text_mode()}));
      cc->OutputSidePackets().Get(output_id).Set(
          MakePacket<std::string>(std::move(contents)));
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(LocalFileContentsCalculator);

}  // namespace mediapipe
