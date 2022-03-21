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

constexpr char kContentsTag[] = "CONTENTS";
constexpr char kFileSuffixTag[] = "FILE_SUFFIX";
constexpr char kFileDirectoryTag[] = "FILE_DIRECTORY";

// The calculator takes the path to local directory and desired file suffix to
// mach as input side packets, and outputs the contents of those files that
// match the pattern. Those matched files will be sent sequentially through the
// output stream with incremental timestamp difference by 1.
//
// Example config:
// node {
//   calculator: "LocalFilePatternContentsCalculator"
//   input_side_packet: "FILE_DIRECTORY:file_directory"
//   input_side_packet: "FILE_SUFFIX:file_suffix"
//   output_stream: "CONTENTS:contents"
// }
class LocalFilePatternContentsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Tag(kFileDirectoryTag).Set<std::string>();
    cc->InputSidePackets().Tag(kFileSuffixTag).Set<std::string>();
    cc->Outputs().Tag(kContentsTag).Set<std::string>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    MP_RETURN_IF_ERROR(mediapipe::file::MatchFileTypeInDirectory(
        cc->InputSidePackets().Tag(kFileDirectoryTag).Get<std::string>(),
        cc->InputSidePackets().Tag(kFileSuffixTag).Get<std::string>(),
        &filenames_));
    std::sort(filenames_.begin(), filenames_.end());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (current_output_ < filenames_.size()) {
      auto contents = absl::make_unique<std::string>();
      LOG(INFO) << filenames_[current_output_];
      MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
          filenames_[current_output_], contents.get()));
      ++current_output_;
      cc->Outputs()
          .Tag(kContentsTag)
          .Add(contents.release(), Timestamp(current_output_));
    } else {
      return tool::StatusStop();
    }
    return absl::OkStatus();
  }

 private:
  std::vector<std::string> filenames_;
  int current_output_ = 0;
};

REGISTER_CALCULATOR(LocalFilePatternContentsCalculator);

}  // namespace mediapipe
