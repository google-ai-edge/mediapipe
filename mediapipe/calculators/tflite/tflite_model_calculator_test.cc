// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT
#include "tensorflow/lite/model.h"

namespace mediapipe {

TEST(TfLiteModelCalculatorTest, SmokeTest) {
  // Prepare single calculator graph to and wait for packets.
  CalculatorGraphConfig graph_config = ParseTextProtoOrDie<
      CalculatorGraphConfig>(
      R"(
        node {
          calculator: "ConstantSidePacketCalculator"
          output_side_packet: "PACKET:model_path"
          options: {
            [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
              packet {
                string_value: "mediapipe/calculators/tflite/testdata/add.bin"
              }
            }
          }
        }

        node {
          calculator: "LocalFileContentsCalculator"
          input_side_packet: "FILE_PATH:model_path"
          output_side_packet: "CONTENTS:model_blob"
        }

        node {
          calculator: "TfLiteModelCalculator"
          input_side_packet: "MODEL_BLOB:model_blob"
          output_side_packet: "MODEL:model"
        }
      )");
  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  auto status_or_packet = graph.GetOutputSidePacket("model");
  MP_ASSERT_OK(status_or_packet);
  auto model_packet = status_or_packet.ValueOrDie();
  const auto& model = model_packet.Get<
      std::unique_ptr<tflite::FlatBufferModel,
                      std::function<void(tflite::FlatBufferModel*)>>>();

  auto expected_model = tflite::FlatBufferModel::BuildFromFile(
      "mediapipe/calculators/tflite/testdata/add.bin");

  EXPECT_EQ(model->GetModel()->version(),
            expected_model->GetModel()->version());
  EXPECT_EQ(model->GetModel()->buffers()->size(),
            expected_model->GetModel()->buffers()->size());
  const int num_subgraphs = expected_model->GetModel()->subgraphs()->size();
  EXPECT_EQ(model->GetModel()->subgraphs()->size(), num_subgraphs);
  for (int i = 0; i < num_subgraphs; ++i) {
    const auto* expected_subgraph =
        expected_model->GetModel()->subgraphs()->Get(i);
    const auto* subgraph = model->GetModel()->subgraphs()->Get(i);
    const int num_tensors = expected_subgraph->tensors()->size();
    EXPECT_EQ(subgraph->tensors()->size(), num_tensors);
    for (int j = 0; j < num_tensors; ++j) {
      EXPECT_EQ(subgraph->tensors()->Get(j)->name()->str(),
                expected_subgraph->tensors()->Get(j)->name()->str());
    }
  }
}

}  // namespace mediapipe
