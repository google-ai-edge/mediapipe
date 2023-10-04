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
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"

namespace mediapipe {
namespace {

using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::testing::ElementsAreArray;

constexpr int kNumInputTensorsForBert = 3;
constexpr int kBertMaxSeqLen = 128;
constexpr absl::string_view kTestModelPath =
    "mediapipe/tasks/testdata/text/bert_text_classifier.tflite";

absl::StatusOr<std::vector<std::vector<int>>> RunBertPreprocessorCalculator(
    absl::string_view text, absl::string_view model_path,
    bool has_dynamic_input_tensors = false, int tensor_size = kBertMaxSeqLen) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      absl::Substitute(R"(
        input_stream: "text"
        output_stream: "tensors"
        node {
          calculator: "BertPreprocessorCalculator"
          input_stream: "TEXT:text"
          input_side_packet: "METADATA_EXTRACTOR:metadata_extractor"
          output_stream: "TENSORS:tensors"
          options {
            [mediapipe.BertPreprocessorCalculatorOptions.ext] {
              bert_max_seq_len: $0
              has_dynamic_input_tensors: $1
            }
          }
        }
      )",
                       tensor_size, has_dynamic_input_tensors));
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensors", &graph_config, &output_packets);

  std::string model_buffer = tasks::core::LoadBinaryContent(model_path.data());
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
  if (tensor_vec.size() != kNumInputTensorsForBert) {
    return absl::InvalidArgumentError(
        absl::Substitute("tensor_vec has size $0, expected $1",
                         tensor_vec.size(), kNumInputTensorsForBert));
  }

  std::vector<std::vector<int>> results;
  for (int i = 0; i < tensor_vec.size(); i++) {
    const Tensor& tensor = tensor_vec[i];
    if (tensor.element_type() != Tensor::ElementType::kInt32) {
      return absl::InvalidArgumentError("Expected tensor element type kInt32");
    }
    auto* buffer = tensor.GetCpuReadView().buffer<int>();
    std::vector<int> buffer_view(buffer, buffer + tensor_size);
    results.push_back(buffer_view);
  }
  MP_RETURN_IF_ERROR(graph.CloseAllPacketSources());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return results;
}

TEST(BertPreprocessorCalculatorTest, TextClassifierWithBertModel) {
  std::vector<std::vector<int>> expected_result = {
      {101, 2009, 1005, 1055, 1037, 11951, 1998, 2411, 12473, 4990, 102}};
  // segment_ids
  expected_result.push_back(std::vector(kBertMaxSeqLen, 0));
  // input_masks
  expected_result.push_back(std::vector(expected_result[0].size(), 1));
  expected_result[2].resize(kBertMaxSeqLen);
  // padding input_ids
  expected_result[0].resize(kBertMaxSeqLen);

  MP_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<int>> processed_tensor_values,
      RunBertPreprocessorCalculator(
          "it's a charming and often affecting journey", kTestModelPath));
  EXPECT_THAT(processed_tensor_values, ElementsAreArray(expected_result));
}

TEST(BertPreprocessorCalculatorTest, LongInput) {
  std::stringstream long_input;
  long_input
      << "it's a charming and often affecting journey and this is a long";
  for (int i = 0; i < kBertMaxSeqLen; ++i) {
    long_input << " long";
  }
  long_input << " movie review";
  std::vector<std::vector<int>> expected_result = {
      {101, 2009, 1005, 1055, 1037, 11951, 1998, 2411, 12473, 4990, 1998, 2023,
       2003, 1037}};
  // "long" id
  expected_result[0].resize(kBertMaxSeqLen - 1, 2146);
  // "[SEP]" id
  expected_result[0].push_back(102);
  // segment_ids
  expected_result.push_back(std::vector(kBertMaxSeqLen, 0));
  // input_masks
  expected_result.push_back(std::vector(kBertMaxSeqLen, 1));

  MP_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<int>> processed_tensor_values,
      RunBertPreprocessorCalculator(long_input.str(), kTestModelPath));
  EXPECT_THAT(processed_tensor_values, ElementsAreArray(expected_result));
}

}  // namespace
}  // namespace mediapipe
