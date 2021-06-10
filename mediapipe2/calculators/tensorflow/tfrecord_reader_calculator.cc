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
#include <utility>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

namespace mediapipe {

const char kTFRecordPath[] = "TFRECORD_PATH";
const char kRecordIndex[] = "RECORD_INDEX";
const char kExampleTag[] = "EXAMPLE";
const char kSequenceExampleTag[] = "SEQUENCE_EXAMPLE";

// Reads a tensorflow example/sequence example from a tfrecord file.
// If the "RECORD_INDEX" input side packet is provided, the calculator is going
// to fetch the example/sequence example of the tfrecord file at the target
// record index. Otherwise, the reader always reads the first example/sequence
// example of the tfrecord file.
//
// Example config:
// node {
//   calculator: "TFRecordReaderCalculator"
//   input_side_packet: "TFRECORD_PATH:tfrecord_path"
//   input_side_packet: "RECORD_INDEX:record_index"
//   output_side_packet: "SEQUENCE_EXAMPLE:sequence_example"
// }
class TFRecordReaderCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
};

absl::Status TFRecordReaderCalculator::GetContract(CalculatorContract* cc) {
  cc->InputSidePackets().Tag(kTFRecordPath).Set<std::string>();
  if (cc->InputSidePackets().HasTag(kRecordIndex)) {
    cc->InputSidePackets().Tag(kRecordIndex).Set<int>();
  }

  RET_CHECK(cc->OutputSidePackets().HasTag(kExampleTag) ||
            cc->OutputSidePackets().HasTag(kSequenceExampleTag))
      << "TFRecordReaderCalculator must output either Tensorflow example or "
         "sequence example.";
  if (cc->OutputSidePackets().HasTag(kExampleTag)) {
    cc->OutputSidePackets().Tag(kExampleTag).Set<tensorflow::Example>();
  } else {
    cc->OutputSidePackets()
        .Tag(kSequenceExampleTag)
        .Set<tensorflow::SequenceExample>();
  }
  return absl::OkStatus();
}

absl::Status TFRecordReaderCalculator::Open(CalculatorContext* cc) {
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  auto tf_status = tensorflow::Env::Default()->NewRandomAccessFile(
      cc->InputSidePackets().Tag(kTFRecordPath).Get<std::string>(), &file);
  RET_CHECK(tf_status.ok())
      << "Failed to open tfrecord file: " << tf_status.ToString();
  tensorflow::io::RecordReader reader(file.get(),
                                      tensorflow::io::RecordReaderOptions());
  tensorflow::uint64 offset = 0;
  tensorflow::tstring example_str;
  const int target_idx =
      cc->InputSidePackets().HasTag(kRecordIndex)
          ? cc->InputSidePackets().Tag(kRecordIndex).Get<int>()
          : 0;
  int current_idx = 0;
  while (current_idx <= target_idx) {
    tf_status = reader.ReadRecord(&offset, &example_str);
    RET_CHECK(tf_status.ok())
        << "Failed to read tfrecord: " << tf_status.ToString();
    if (current_idx == target_idx) {
      if (cc->OutputSidePackets().HasTag(kExampleTag)) {
        tensorflow::Example tf_example;
        tf_example.ParseFromArray(example_str.data(), example_str.size());
        cc->OutputSidePackets()
            .Tag(kExampleTag)
            .Set(MakePacket<tensorflow::Example>(std::move(tf_example)));
      } else {
        tensorflow::SequenceExample tf_sequence_example;
        tf_sequence_example.ParseFromString(example_str);
        cc->OutputSidePackets()
            .Tag(kSequenceExampleTag)
            .Set(MakePacket<tensorflow::SequenceExample>(
                std::move(tf_sequence_example)));
      }
    }
    ++current_idx;
  }

  return absl::OkStatus();
}

absl::Status TFRecordReaderCalculator::Process(CalculatorContext* cc) {
  return absl::OkStatus();
}

REGISTER_CALCULATOR(TFRecordReaderCalculator);

}  // namespace mediapipe
