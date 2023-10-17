// Copyright 2023 The MediaPipe Authors.
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
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace {

using ::testing::SizeIs;
using ::testing::TestWithParam;
using Options = mediapipe::TensorsToSegmentationCalculatorOptions;

std::string ActivationTypeToString(Options::Activation activation) {
  switch (activation) {
    case Options::NONE:
      return "NONE";
    case Options::SIGMOID:
      return "SIGMOID";
    case Options::SOFTMAX:
      return "SOFTMAX";
    default:
      ABSL_LOG(FATAL) << "Unknown activation type: " << activation;
      return "UNKNOWN";
  }
}

struct FormattingTestCase {
  std::string test_name;
  std::vector<float> inputs;
  std::vector<float> expected_outputs;
  Options::Activation activation;
  int rows;
  int cols;
  int channels;
};

using TensorsToSegmentationCalculatorTest = TestWithParam<FormattingTestCase>;

// Currently only useable for tests with no output resize.
TEST_P(TensorsToSegmentationCalculatorTest, ParameterizedTests) {
  const FormattingTestCase& test_case = GetParam();
  std::vector<float> inputs = test_case.inputs;
  std::vector<float> expected_outputs = test_case.expected_outputs;
  Options::Activation activation = test_case.activation;
  int rows = test_case.rows;
  int cols = test_case.cols;
  int channels = test_case.channels;

  std::string string_config = absl::Substitute(
      R"pb(
        input_stream: "tensors"
        input_stream: "size"
        node {
          calculator: "TensorsToSegmentationCalculator"
          input_stream: "TENSORS:tensors"
          input_stream: "OUTPUT_SIZE:size"
          output_stream: "MASK:image_as_mask"
          options: {
            [mediapipe.TensorsToSegmentationCalculatorOptions.ext] {
              activation: $0
            }
          }
        }
      )pb",
      ActivationTypeToString(activation));
  auto graph_config =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(string_config);

  std::vector<Packet> output_packets;
  tool::AddVectorSink("image_as_mask", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  auto tensors = std::make_unique<std::vector<Tensor>>();
  tensors->emplace_back(Tensor::ElementType::kFloat32,
                        Tensor::Shape{1, rows, cols, channels});

  // We scope the tensor's GetCpuWriteView() call so that its lock is released
  // before we pass it into the graph.
  {
    auto view = tensors->back().GetCpuWriteView();
    float* tensor_buffer = view.buffer<float>();
    for (int i = 0; i < inputs.size(); ++i) {
      tensor_buffer[i] = inputs[i];
    }
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "tensors", mediapipe::Adopt(tensors.release()).At(Timestamp(0))));
  }
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "size",
      mediapipe::Adopt(new std::pair<int, int>(rows, cols)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  ASSERT_THAT(output_packets, SizeIs(1));
  const Image& image_as_mask = output_packets[0].Get<Image>();
  std::shared_ptr<cv::Mat> result_mat = formats::MatView(&image_as_mask);
  EXPECT_EQ(result_mat->rows, rows);
  EXPECT_EQ(result_mat->cols, cols);
  EXPECT_EQ(result_mat->channels(), channels);

  // Compare the real result with the expected result.
  cv::Mat expected_result = cv::Mat(
      rows, cols, CV_32FC1, const_cast<float*>(expected_outputs.data()));
  cv::Mat diff;
  cv::absdiff(*result_mat, expected_result, diff);
  double max_val;
  cv::minMaxLoc(diff, nullptr, &max_val);
  // Expects the maximum absolute pixel-by-pixel difference is less than 1e-5.
  // This delta is for passthorugh accuracy only.
  EXPECT_LE(max_val, 1e-5);

  MP_ASSERT_OK(graph.CloseInputStream("tensors"));
  MP_ASSERT_OK(graph.CloseInputStream("size"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

INSTANTIATE_TEST_SUITE_P(
    TensorsToSegmentationCalculatorTests, TensorsToSegmentationCalculatorTest,
    testing::ValuesIn<FormattingTestCase>({
        {/*test_name=*/"NoActivationAndNoOutputResize",
         /*inputs=*/
         {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
          14.0, 15.0, 16.0},
         /*expected_outputs=*/
         {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
          14.0, 15.0, 16.0},
         /*activation=*/Options::NONE,
         /*rows=*/4,
         /*cols=*/4,
         /*channels=*/1},
    }),
    [](const testing::TestParamInfo<
        TensorsToSegmentationCalculatorTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace mediapipe
