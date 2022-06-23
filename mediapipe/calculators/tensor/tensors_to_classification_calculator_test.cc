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

#include <algorithm>
#include <limits>
#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

using mediapipe::ParseTextProtoOrDie;
using Node = ::mediapipe::CalculatorGraphConfig::Node;

class TensorsToClassificationCalculatorTest : public ::testing::Test {
 protected:
  void BuildGraph(mediapipe::CalculatorRunner* runner,
                  const std::vector<float>& scores) {
    auto tensors = absl::make_unique<std::vector<Tensor>>();
    tensors->emplace_back(
        Tensor::ElementType::kFloat32,
        Tensor::Shape{1, 1, static_cast<int>(scores.size()), 1});
    auto view = tensors->back().GetCpuWriteView();
    float* tensor_buffer = view.buffer<float>();
    ASSERT_NE(tensor_buffer, nullptr);
    for (int i = 0; i < scores.size(); ++i) {
      tensor_buffer[i] = scores[i];
    }

    int64 stream_timestamp = 0;
    auto& input_stream_packets =
        runner->MutableInputs()->Tag("TENSORS").packets;

    input_stream_packets.push_back(
        mediapipe::Adopt(tensors.release())
            .At(mediapipe::Timestamp(stream_timestamp++)));
  }
};

TEST_F(TensorsToClassificationCalculatorTest, CorrectOutput) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToClassificationCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "CLASSIFICATIONS:classifications"
    options {
      [mediapipe.TensorsToClassificationCalculatorOptions.ext] {}
    }
  )pb"));

  BuildGraph(&runner, {0, 0.5, 1});
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets_ = runner.Outputs().Tag("CLASSIFICATIONS").packets;

  EXPECT_EQ(1, output_packets_.size());

  const auto& classification_list =
      output_packets_[0].Get<ClassificationList>();
  EXPECT_EQ(3, classification_list.classification_size());

  // Verify that the label_id and score fields are set correctly.
  for (int i = 0; i < classification_list.classification_size(); ++i) {
    EXPECT_EQ(i, classification_list.classification(i).index());
    EXPECT_EQ(i * 0.5, classification_list.classification(i).score());
    ASSERT_FALSE(classification_list.classification(i).has_label());
  }
}

TEST_F(TensorsToClassificationCalculatorTest, CorrectOutputWithLabelMapPath) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToClassificationCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "CLASSIFICATIONS:classifications"
    options {
      [mediapipe.TensorsToClassificationCalculatorOptions.ext] {
        label_map_path: "mediapipe/calculators/tensor/testdata/labelmap.txt"
      }
    }
  )pb"));

  BuildGraph(&runner, {0, 0.5, 1});
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets_ = runner.Outputs().Tag("CLASSIFICATIONS").packets;

  EXPECT_EQ(1, output_packets_.size());

  const auto& classification_list =
      output_packets_[0].Get<ClassificationList>();
  EXPECT_EQ(3, classification_list.classification_size());

  // Verify that the label field is set.
  for (int i = 0; i < classification_list.classification_size(); ++i) {
    EXPECT_EQ(i, classification_list.classification(i).index());
    EXPECT_EQ(i * 0.5, classification_list.classification(i).score());
    ASSERT_TRUE(classification_list.classification(i).has_label());
  }
}

TEST_F(TensorsToClassificationCalculatorTest, CorrectOutputWithLabelMap) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToClassificationCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "CLASSIFICATIONS:classifications"
    options {
      [mediapipe.TensorsToClassificationCalculatorOptions.ext] {
        label_map {
          entries { id: 0, label: "ClassA" }
          entries { id: 1, label: "ClassB" }
          entries { id: 2, label: "ClassC" }
        }
      }
    }
  )pb"));

  BuildGraph(&runner, {0, 0.5, 1});
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets_ = runner.Outputs().Tag("CLASSIFICATIONS").packets;

  EXPECT_EQ(1, output_packets_.size());

  const auto& classification_list =
      output_packets_[0].Get<ClassificationList>();
  EXPECT_EQ(3, classification_list.classification_size());

  // Verify that the label field is set.
  for (int i = 0; i < classification_list.classification_size(); ++i) {
    EXPECT_EQ(i, classification_list.classification(i).index());
    EXPECT_EQ(i * 0.5, classification_list.classification(i).score());
    ASSERT_TRUE(classification_list.classification(i).has_label());
  }
}

TEST_F(TensorsToClassificationCalculatorTest,
       CorrectOutputWithLabelMinScoreThreshold) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToClassificationCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "CLASSIFICATIONS:classifications"
    options {
      [mediapipe.TensorsToClassificationCalculatorOptions.ext] {
        min_score_threshold: 0.6
      }
    }
  )pb"));

  BuildGraph(&runner, {0, 0.5, 1});
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets_ = runner.Outputs().Tag("CLASSIFICATIONS").packets;

  EXPECT_EQ(1, output_packets_.size());

  const auto& classification_list =
      output_packets_[0].Get<ClassificationList>();

  // Verify that the low score labels are filtered out.
  EXPECT_EQ(1, classification_list.classification_size());
  EXPECT_EQ(1, classification_list.classification(0).score());
}

TEST_F(TensorsToClassificationCalculatorTest, CorrectOutputWithTopK) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToClassificationCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "CLASSIFICATIONS:classifications"
    options {
      [mediapipe.TensorsToClassificationCalculatorOptions.ext] { top_k: 2 }
    }
  )pb"));

  BuildGraph(&runner, {0, 0.5, 1});
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets_ = runner.Outputs().Tag("CLASSIFICATIONS").packets;

  EXPECT_EQ(1, output_packets_.size());

  const auto& classification_list =
      output_packets_[0].Get<ClassificationList>();

  // Verify that the only top2 labels are left.
  EXPECT_EQ(2, classification_list.classification_size());
  for (int i = 0; i < classification_list.classification_size(); ++i) {
    EXPECT_EQ((classification_list.classification_size() - i) * 0.5,
              classification_list.classification(i).score());
  }
}

TEST_F(TensorsToClassificationCalculatorTest,
       CorrectOutputWithSortByDescendingScore) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToClassificationCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "CLASSIFICATIONS:classifications"
    options {
      [mediapipe.TensorsToClassificationCalculatorOptions.ext] {
        sort_by_descending_score: true
      }
    }
  )pb"));

  BuildGraph(&runner, {0, 0.5, 1});
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets_ = runner.Outputs().Tag("CLASSIFICATIONS").packets;

  EXPECT_EQ(1, output_packets_.size());

  const auto& classification_list =
      output_packets_[0].Get<ClassificationList>();

  // Verify results are sorted by descending score.
  EXPECT_EQ(3, classification_list.classification_size());
  float score = std::numeric_limits<float>::max();
  for (int i = 0; i < classification_list.classification_size(); ++i) {
    EXPECT_LE(classification_list.classification(i).score(), score);
    score = classification_list.classification(i).score();
  }
}

TEST_F(TensorsToClassificationCalculatorTest,
       ClassNameAllowlistWithLabelItems) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToClassificationCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "CLASSIFICATIONS:classifications"
    options {
      [mediapipe.TensorsToClassificationCalculatorOptions.ext] {
        label_items {
          key: 0
          value { name: "ClassA" }
        }
        label_items {
          key: 1
          value { name: "ClassB" }
        }
        label_items {
          key: 2
          value { name: "ClassC" }
        }
        allow_classes: 1
      }
    }
  )pb"));

  BuildGraph(&runner, {0, 0.5, 1});
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets_ = runner.Outputs().Tag("CLASSIFICATIONS").packets;

  EXPECT_EQ(1, output_packets_.size());

  const auto& classification_list =
      output_packets_[0].Get<ClassificationList>();
  EXPECT_EQ(1, classification_list.classification_size());
  EXPECT_EQ(1, classification_list.classification(0).index());
  EXPECT_EQ(0.5, classification_list.classification(0).score());
  ASSERT_TRUE(classification_list.classification(0).has_label());
}

TEST_F(TensorsToClassificationCalculatorTest,
       ClassNameIgnorelistWithLabelItems) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToClassificationCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "CLASSIFICATIONS:classifications"
    options {
      [mediapipe.TensorsToClassificationCalculatorOptions.ext] {
        label_items {
          key: 0
          value { name: "ClassA" }
        }
        label_items {
          key: 1
          value { name: "ClassB" }
        }
        label_items {
          key: 2
          value { name: "ClassC" }
        }
        ignore_classes: 1
      }
    }
  )pb"));

  BuildGraph(&runner, {0, 0.5, 1});
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets_ = runner.Outputs().Tag("CLASSIFICATIONS").packets;

  EXPECT_EQ(1, output_packets_.size());

  const auto& classification_list =
      output_packets_[0].Get<ClassificationList>();
  EXPECT_EQ(2, classification_list.classification_size());
  EXPECT_EQ(0, classification_list.classification(0).index());
  EXPECT_EQ(0, classification_list.classification(0).score());
  ASSERT_TRUE(classification_list.classification(0).has_label());
  EXPECT_EQ(2, classification_list.classification(1).index());
  EXPECT_EQ(1, classification_list.classification(1).score());
  ASSERT_TRUE(classification_list.classification(1).has_label());
}

}  // namespace mediapipe
