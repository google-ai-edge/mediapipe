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
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"

namespace mediapipe {
namespace {

using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;
using ::testing::HasSubstr;
using Node = ::mediapipe::CalculatorGraphConfig::Node;

// Builds the graph and feeds inputs.
void BuildGraph(CalculatorRunner* runner,
                std::vector<std::vector<float>> tensors) {
  auto inputs = std::make_unique<std::vector<Tensor>>();
  for (const auto& tensor : tensors) {
    inputs->emplace_back(Tensor::ElementType::kFloat32,
                         Tensor::Shape{1, static_cast<int>(tensor.size())});
    auto view = inputs->back().GetCpuWriteView();
    float* buffer = view.buffer<float>();
    ASSERT_NE(buffer, nullptr);
    for (int i = 0; i < tensor.size(); ++i) {
      buffer[i] = tensor[i];
    }
  }
  auto& input_packets = runner->MutableInputs()->Tag("TENSORS").packets;
  input_packets.push_back(Adopt(inputs.release()).At(Timestamp(0)));
}

TEST(TensorsToEmbeddingsCalculatorTest, FailsWithInvalidHeadNamesNumber) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToEmbeddingsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "EMBEDDINGS:embeddings"
    options {
      [mediapipe.TensorsToEmbeddingsCalculatorOptions.ext] { head_names: "foo" }
    }
  )pb"));

  BuildGraph(&runner, {{0.1, 0.2}, {0.2, 0.3}});
  auto status = runner.Run();

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Mismatch between number of provided head names"));
}

TEST(TensorsToEmbeddingsCalculatorTest, SucceedsWithoutHeadNames) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToEmbeddingsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "EMBEDDINGS:embeddings"
    options {
      [mediapipe.TensorsToEmbeddingsCalculatorOptions.ext] {
        embedder_options { l2_normalize: false quantize: false }
      }
    }
  )pb"));

  BuildGraph(&runner, {{0.1, 0.2}, {-0.2, -0.3}});
  MP_ASSERT_OK(runner.Run());

  const EmbeddingResult& result =
      runner.Outputs().Get("EMBEDDINGS", 0).packets[0].Get<EmbeddingResult>();
  EXPECT_THAT(result, EqualsProto(ParseTextProtoOrDie<EmbeddingResult>(
                          R"pb(embeddings {
                                 float_embedding { values: 0.1 values: 0.2 }
                                 head_index: 0
                               }
                               embeddings {
                                 float_embedding { values: -0.2 values: -0.3 }
                                 head_index: 1
                               })pb")));
}

TEST(TensorsToEmbeddingsCalculatorTest, SucceedsWithHeadNames) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToEmbeddingsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "EMBEDDINGS:embeddings"
    options {
      [mediapipe.TensorsToEmbeddingsCalculatorOptions.ext] {
        embedder_options { l2_normalize: false quantize: false }
        head_names: "foo"
        head_names: "bar"
      }
    }
  )pb"));

  BuildGraph(&runner, {{0.1, 0.2}, {-0.2, -0.3}});
  MP_ASSERT_OK(runner.Run());

  const EmbeddingResult& result =
      runner.Outputs().Get("EMBEDDINGS", 0).packets[0].Get<EmbeddingResult>();
  EXPECT_THAT(result, EqualsProto(ParseTextProtoOrDie<EmbeddingResult>(
                          R"pb(embeddings {
                                 float_embedding { values: 0.1 values: 0.2 }
                                 head_index: 0
                                 head_name: "foo"
                               }
                               embeddings {
                                 float_embedding { values: -0.2 values: -0.3 }
                                 head_index: 1
                                 head_name: "bar"
                               })pb")));
}

TEST(TensorsToEmbeddingsCalculatorTest, SucceedsWithHeadNameIgnored) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToEmbeddingsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "EMBEDDINGS:embeddings"
    options {
      [mediapipe.TensorsToEmbeddingsCalculatorOptions.ext] {
        embedder_options { l2_normalize: false quantize: false }
        head_names: "foo"
        head_names: "bar"
        ignored_head_names: "foo"
      }
    }
  )pb"));

  BuildGraph(&runner, {{0.1, 0.2}, {-0.2, -0.3}});
  MP_ASSERT_OK(runner.Run());

  const EmbeddingResult& result =
      runner.Outputs().Get("EMBEDDINGS", 0).packets[0].Get<EmbeddingResult>();
  EXPECT_THAT(result, EqualsProto(ParseTextProtoOrDie<EmbeddingResult>(
                          R"pb(
                            embeddings {
                              float_embedding { values: -0.2 values: -0.3 }
                              head_index: 1
                              head_name: "bar"
                            })pb")));
}

TEST(TensorsToEmbeddingsCalculatorTest, SucceedsWithBothHeadsIgnored) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToEmbeddingsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "EMBEDDINGS:embeddings"
    options {
      [mediapipe.TensorsToEmbeddingsCalculatorOptions.ext] {
        embedder_options { l2_normalize: false quantize: false }
        head_names: "foo"
        head_names: "bar"
        ignored_head_names: "foo"
        ignored_head_names: "bar"
      }
    }
  )pb"));

  BuildGraph(&runner, {{0.1, 0.2}, {-0.2, -0.3}});
  MP_ASSERT_OK(runner.Run());

  const EmbeddingResult& result =
      runner.Outputs().Get("EMBEDDINGS", 0).packets[0].Get<EmbeddingResult>();
  EXPECT_THAT(result,
              EqualsProto(ParseTextProtoOrDie<EmbeddingResult>(R"pb()pb")));
}

TEST(TensorsToEmbeddingsCalculatorTest, SucceedsWithNormalization) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToEmbeddingsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "EMBEDDINGS:embeddings"
    options {
      [mediapipe.TensorsToEmbeddingsCalculatorOptions.ext] {
        embedder_options { l2_normalize: true quantize: false }
      }
    }
  )pb"));

  BuildGraph(&runner, {{0.1, 0.2}, {-0.2, -0.3}});
  MP_ASSERT_OK(runner.Run());

  const EmbeddingResult& result =
      runner.Outputs().Get("EMBEDDINGS", 0).packets[0].Get<EmbeddingResult>();
  EXPECT_THAT(
      result,
      EqualsProto(ParseTextProtoOrDie<EmbeddingResult>(
          R"pb(embeddings {
                 float_embedding { values: 0.44721356 values: 0.8944271 }
                 head_index: 0
               }
               embeddings {
                 float_embedding { values: -0.5547002 values: -0.8320503 }
                 head_index: 1
               })pb")));
}

TEST(TensorsToEmbeddingsCalculatorTest, SucceedsWithQuantization) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToEmbeddingsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "EMBEDDINGS:embeddings"
    options {
      [mediapipe.TensorsToEmbeddingsCalculatorOptions.ext] {
        embedder_options { l2_normalize: false quantize: true }
      }
    }
  )pb"));

  BuildGraph(&runner, {{0.1, 0.2}, {-0.2, -0.3}});
  MP_ASSERT_OK(runner.Run());

  const EmbeddingResult& result =
      runner.Outputs().Get("EMBEDDINGS", 0).packets[0].Get<EmbeddingResult>();
  EXPECT_THAT(result,
              EqualsProto(ParseTextProtoOrDie<EmbeddingResult>(
                  R"pb(embeddings {
                         quantized_embedding { values: "\x0d\x1a" }  # 13,26
                         head_index: 0
                       }
                       embeddings {
                         quantized_embedding { values: "\xe6\xda" }  # -26,-38
                         head_index: 1
                       })pb")));
}

TEST(TensorsToEmbeddingsCalculatorTest,
     SucceedsWithNormalizationAndQuantization) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToEmbeddingsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "EMBEDDINGS:embeddings"
    options {
      [mediapipe.TensorsToEmbeddingsCalculatorOptions.ext] {
        embedder_options { l2_normalize: true quantize: true }
      }
    }
  )pb"));

  BuildGraph(&runner, {{0.1, 0.2}, {-0.2, -0.3}});
  MP_ASSERT_OK(runner.Run());

  const EmbeddingResult& result =
      runner.Outputs().Get("EMBEDDINGS", 0).packets[0].Get<EmbeddingResult>();
  EXPECT_THAT(result,
              EqualsProto(ParseTextProtoOrDie<EmbeddingResult>(
                  R"pb(embeddings {
                         quantized_embedding { values: "\x39\x72" }  # 57,114
                         head_index: 0
                       }
                       embeddings {
                         quantized_embedding { values: "\xb9\x95" }  # -71,-107
                         head_index: 1
                       })pb")));
}

}  // namespace
}  // namespace mediapipe
