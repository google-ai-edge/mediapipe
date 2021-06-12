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

#include "absl/flags/flag.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

std::string GetGoldenFilePath(const std::string& filename) {
  return mediapipe::file::JoinPath(
      "./", "mediapipe/calculators/tflite/testdata/" + filename);
}

void ParseAnchorsFromText(const std::string& text,
                          std::vector<Anchor>* anchors) {
  const std::string line_delimiter = "\n";
  const std::string number_delimiter = ",";

  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    Anchor anchor;
    float values[4];
    std::string::size_type pos;
    for (int i = 0; i < 4; ++i) {
      values[i] = std::stof(line, &pos);
      line = line.substr(pos);
    }
    anchor.set_x_center(values[0]);
    anchor.set_y_center(values[1]);
    anchor.set_w(values[2]);
    anchor.set_h(values[3]);
    anchors->push_back(anchor);
  }
}

void CompareAnchors(const std::vector<Anchor>& anchors_0,
                    const std::vector<Anchor>& anchors_1) {
  EXPECT_EQ(anchors_0.size(), anchors_1.size());
  for (int i = 0; i < anchors_0.size(); ++i) {
    const auto& anchor_0 = anchors_0[i];
    const auto& anchor_1 = anchors_1[i];
    EXPECT_THAT(anchor_0.x_center(),
                testing::FloatNear(anchor_1.x_center(), 1e-5));
    EXPECT_THAT(anchor_0.y_center(),
                testing::FloatNear(anchor_1.y_center(), 1e-5));
  }
}

TEST(SsdAnchorCalculatorTest, FaceDetectionConfig) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "SsdAnchorsCalculator"
    output_side_packet: "anchors"
    options {
      [mediapipe.SsdAnchorsCalculatorOptions.ext] {
        num_layers: 5
        min_scale: 0.1171875
        max_scale: 0.75
        input_size_height: 256
        input_size_width: 256
        anchor_offset_x: 0.5
        anchor_offset_y: 0.5
        strides: 8
        strides: 16
        strides: 32
        strides: 32
        strides: 32
        aspect_ratios: 1.0
        fixed_anchor_size: true
      }
    }
  )pb"));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";

  const auto& anchors =
      runner.OutputSidePackets().Index(0).Get<std::vector<Anchor>>();
  std::string anchors_string;
  MP_EXPECT_OK(mediapipe::file::GetContents(
      GetGoldenFilePath("anchor_golden_file_0.txt"), &anchors_string));

  std::vector<Anchor> anchors_golden;
  ParseAnchorsFromText(anchors_string, &anchors_golden);

  CompareAnchors(anchors, anchors_golden);
}

TEST(SsdAnchorCalculatorTest, MobileSSDConfig) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "SsdAnchorsCalculator"
    output_side_packet: "anchors"
    options {
      [mediapipe.SsdAnchorsCalculatorOptions.ext] {
        num_layers: 6
        min_scale: 0.2
        max_scale: 0.95
        input_size_height: 300
        input_size_width: 300
        anchor_offset_x: 0.5
        anchor_offset_y: 0.5
        strides: 16
        strides: 32
        strides: 64
        strides: 128
        strides: 256
        strides: 512
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        aspect_ratios: 3.0
        aspect_ratios: 0.3333
        reduce_boxes_in_lowest_layer: true
      }
    }
  )pb"));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const auto& anchors =
      runner.OutputSidePackets().Index(0).Get<std::vector<Anchor>>();

  std::string anchors_string;
  MP_EXPECT_OK(mediapipe::file::GetContents(
      GetGoldenFilePath("anchor_golden_file_1.txt"), &anchors_string));

  std::vector<Anchor> anchors_golden;
  ParseAnchorsFromText(anchors_string, &anchors_golden);

  CompareAnchors(anchors, anchors_golden);
}

}  // namespace mediapipe
