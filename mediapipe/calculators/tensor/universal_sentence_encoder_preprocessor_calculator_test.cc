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
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/options_map.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"

namespace mediapipe {
namespace {

using ::mediapipe::IsOkAndHolds;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::testing::ElementsAreArray;

constexpr int kNumInputTensorsForUniversalSentenceEncoder = 3;

constexpr absl::string_view kTestModelPath =
    "mediapipe/tasks/testdata/text/"
    "universal_sentence_encoder_qa_with_metadata.tflite";

absl::StatusOr<std::vector<std::string>>
RunUniversalSentenceEncoderPreprocessorCalculator(absl::string_view text) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "text"
    output_stream: "tensors"
    node {
      calculator: "UniversalSentenceEncoderPreprocessorCalculator"
      input_stream: "TEXT:text"
      input_side_packet: "METADATA_EXTRACTOR:metadata_extractor"
      output_stream: "TENSORS:tensors"
    }
  )pb");
  std::vector<Packet> output_packets;
  tool::AddVectorSink("tensors", &graph_config, &output_packets);

  std::string model_buffer =
      tasks::core::LoadBinaryContent(kTestModelPath.data());
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
  if (tensor_vec.size() != kNumInputTensorsForUniversalSentenceEncoder) {
    return absl::InvalidArgumentError(absl::Substitute(
        "tensor_vec has size $0, expected $1", tensor_vec.size(),
        kNumInputTensorsForUniversalSentenceEncoder));
  }
  if (tensor_vec[0].element_type() != Tensor::ElementType::kChar) {
    return absl::InvalidArgumentError("Expected tensor element type kChar");
  }
  std::vector<std::string> results;
  for (int i = 0; i < kNumInputTensorsForUniversalSentenceEncoder; ++i) {
    results.push_back(
        {tensor_vec[i].GetCpuReadView().buffer<char>(),
         static_cast<size_t>(tensor_vec[i].shape().num_elements())});
  }
  return results;
}

TEST(UniversalSentenceEncoderPreprocessorCalculatorTest, TestUSE) {
  ASSERT_THAT(
      RunUniversalSentenceEncoderPreprocessorCalculator("test_input_text"),
      IsOkAndHolds(ElementsAreArray({"", "", "test_input_text"})));
}

}  // namespace
}  // namespace mediapipe
