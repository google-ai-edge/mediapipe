// Copyright 2022 The MediaPipe Authors.
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
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/sink.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"

namespace mediapipe {
namespace {

using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::testing::ElementsAreArray;

constexpr int kMaxSeqLen = 256;
constexpr char kTestModelPath[] =
    "mediapipe/tasks/testdata/text/"
    "test_model_text_classifier_with_regex_tokenizer.tflite";

absl::StatusOr<std::vector<int>> RunRegexPreprocessorCalculator(
    absl::string_view text) {
  auto graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
          R"pb(
            input_stream: "text"
            output_stream: "tensors"
            node {
              calculator: "RegexPreprocessorCalculator"
              input_stream: "TEXT:text"
              input_side_packet: "METADATA_EXTRACTOR:metadata_extractor"
              output_stream: "TENSORS:tensors"
              options {
                [mediapipe.RegexPreprocessorCalculatorOptions.ext] {
                  max_seq_len: $0
                }
              }
            }
          )pb",
          kMaxSeqLen));
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensors", &graph_config, &output_packets);

  std::string model_buffer = tasks::core::LoadBinaryContent(kTestModelPath);
  MP_ASSIGN_OR_RETURN(
      std::unique_ptr<ModelMetadataExtractor> metadata_extractor,
      ModelMetadataExtractor::CreateFromModelBuffer(model_buffer.data(),
                                                    model_buffer.size()));
  // Run the graph.
  CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(
      graph_config,
      {{"metadata_extractor",
        MakePacket<ModelMetadataExtractor>(std::move(*metadata_extractor))}}));
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "text", MakePacket<std::string>(text).At(Timestamp(0))));
  MP_RETURN_IF_ERROR(graph.WaitUntilIdle());

  if (output_packets.size() != 1) {
    return absl::InvalidArgumentError(absl::Substitute(
        "output_packets has size $0, expected 1", output_packets.size()));
  }
  const std::vector<Tensor>& tensor_vec =
      output_packets[0].Get<std::vector<Tensor>>();
  if (tensor_vec.size() != 1) {
    return absl::InvalidArgumentError(absl::Substitute(
        "tensor_vec has size $0, expected $1", tensor_vec.size(), 1));
  }
  if (tensor_vec[0].element_type() != Tensor::ElementType::kInt32) {
    return absl::InvalidArgumentError("Expected tensor element type kInt32");
  }
  auto* buffer = tensor_vec[0].GetCpuReadView().buffer<int>();
  std::vector<int> result(buffer, buffer + kMaxSeqLen);
  MP_RETURN_IF_ERROR(graph.CloseAllPacketSources());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return result;
}

TEST(RegexPreprocessorCalculatorTest, TextClassifierModel) {
  MP_ASSERT_OK_AND_ASSIGN(
      std::vector<int> processed_tensor_values,
      RunRegexPreprocessorCalculator("This is the best movie I’ve seen in "
                                     "recent years. Strongly recommend it!"));
  static const int expected_result[kMaxSeqLen] = {
      1, 2, 9, 4, 118, 20, 2, 2, 110, 11, 1136, 153, 2, 386, 12};
  EXPECT_THAT(processed_tensor_values, ElementsAreArray(expected_result));
}

TEST(RegexPreprocessorCalculatorTest, LongInput) {
  std::stringstream long_input;
  long_input << "This is the best";
  for (int i = 0; i < kMaxSeqLen; ++i) {
    long_input << " best";
  }
  long_input << "movie I’ve seen in recent years. Strongly recommend it!";
  MP_ASSERT_OK_AND_ASSIGN(std::vector<int> processed_tensor_values,
                          RunRegexPreprocessorCalculator(long_input.str()));
  std::vector<int> expected_result = {1, 2, 9, 4, 118};
  // "best" id
  expected_result.resize(kMaxSeqLen, 118);
  EXPECT_THAT(processed_tensor_values, ElementsAreArray(expected_result));
}

}  // namespace
}  // namespace mediapipe
