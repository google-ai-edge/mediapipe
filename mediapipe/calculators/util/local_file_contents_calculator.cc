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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
// The calculator takes the path to the local file as an input side packet and
// outputs the contents of that file.
//
// Example config:
// node {
//   calculator: "LocalFileContentsCalculator"
//   input_side_packet: "FILE_PATH:file_path"
//   output_side_packet: "CONTENTS:contents"
// }
class LocalFileContentsCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Tag("FILE_PATH").Set<std::string>();
    cc->OutputSidePackets().Tag("CONTENTS").Set<std::string>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    std::string contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        cc->InputSidePackets().Tag("FILE_PATH").Get<std::string>(), &contents));
    cc->OutputSidePackets()
        .Tag("CONTENTS")
        .Set(MakePacket<std::string>(std::move(contents)));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }
};

REGISTER_CALCULATOR(LocalFileContentsCalculator);

}  // namespace mediapipe
