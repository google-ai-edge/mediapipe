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

#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator_test_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace {

using ::testing::SizeIs;
using ::testing::TestWithParam;
using Options = mediapipe::TensorsToSegmentationCalculatorOptions;
namespace test_utils = ::mediapipe::tensors_to_segmentation_utils;

using TensorsToSegmentationCalculatorTest =
    TestWithParam<test_utils::FormattingTestCase>;

TEST_P(TensorsToSegmentationCalculatorTest, ParameterizedTests) {
  const auto& [test_name, inputs, expected_outputs, activation, rows, cols,
               rows_new, cols_new, channels, max_abs_diff] = GetParam();

  auto graph_config =
      test_utils::CreateGraphConfigForTest(/*test_gpu=*/false, activation);

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

  // The output size is defined as pair(new_width, new_height).
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "size", mediapipe::Adopt(new std::pair<int, int>(cols_new, rows_new))
                  .At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  ASSERT_THAT(output_packets, SizeIs(1));
  const Image& image_as_mask = output_packets[0].Get<Image>();
  EXPECT_FALSE(image_as_mask.UsesGpu());

  std::shared_ptr<cv::Mat> result_mat = formats::MatView(&image_as_mask);
  EXPECT_EQ(result_mat->rows, rows_new);
  EXPECT_EQ(result_mat->cols, cols_new);
  EXPECT_EQ(result_mat->channels(), 1);

  // Compare the real result with the expected result.
  cv::Mat expected_result =
      cv::Mat(rows_new, cols_new, CV_32FC1,
              const_cast<float*>(expected_outputs.data()));
  cv::Mat diff;
  cv::absdiff(*result_mat, expected_result, diff);
  double max_val;
  cv::minMaxLoc(diff, nullptr, &max_val);

  // The max allowable diff between output and expected output varies between
  // tests.
  EXPECT_LE(max_val, max_abs_diff);

  MP_ASSERT_OK(graph.CloseInputStream("tensors"));
  MP_ASSERT_OK(graph.CloseInputStream("size"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

INSTANTIATE_TEST_SUITE_P(
    TensorsToSegmentationCalculatorTests, TensorsToSegmentationCalculatorTest,
    testing::ValuesIn<test_utils::FormattingTestCase>({
        {.test_name = "NoActivationAndNoOutputResize",
         .inputs = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                    12.0, 13.0, 14.0, 15.0, 16.0},
         .expected_outputs = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                              11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
         .activation = Options::NONE,
         .rows = 4,
         .cols = 4,
         .rows_new = 4,
         .cols_new = 4,
         .channels = 1,
         .max_abs_diff = 1e-7},
        {.test_name = "OutputResizeOnly",
         .inputs = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                    12.0, 13.0, 14.0, 15.0, 16.0},
         .expected_outputs = {1,    1.5,  2.166667,  2.833333,  3.5,  4,
                              3.8,  4.3,  4.966667,  5.633333,  6.3,  6.8,
                              7,    7.5,  8.166667,  8.833333,  9.5,  10,
                              10.2, 10.7, 11.366667, 12.033333, 12.7, 13.2,
                              13,   13.5, 14.166667, 14.833333, 15.5, 16},
         .activation = Options::NONE,
         .rows = 4,
         .cols = 4,
         .rows_new = 5,
         .cols_new = 6,
         .channels = 1,
         .max_abs_diff = 1e-6},
        {.test_name = "SigmoidActivationWithNoOutputResize",
         .inputs = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                    12.0, 13.0, 14.0, 15.0, 16.0},
         .expected_outputs = {0.731059, 0.880797, 0.952574, 0.982014, 0.993307,
                              0.997527, 0.999089, 0.999665, 0.999877, 0.999955,
                              0.999983, 0.999994, 0.999998, 0.999999, 1.0, 1.0},
         .activation = Options::SIGMOID,
         .rows = 4,
         .cols = 4,
         .rows_new = 4,
         .cols_new = 4,
         .channels = 1,
         .max_abs_diff = 1e-6},
        {.test_name = "SigmoidActivationWithOutputResize",
         .inputs = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                    12.0, 13.0, 14.0, 15.0, 16.0},
         .expected_outputs = {0.731059, 0.805928, 0.89276,  0.940611, 0.967294,
                              0.982014, 0.914633, 0.93857,  0.966279, 0.981363,
                              0.989752, 0.994369, 0.996592, 0.997666, 0.998873,
                              0.999404, 0.999683, 0.999829, 0.999913, 0.99994,
                              0.999971, 0.999985, 0.999992, 0.999996, 0.999998,
                              0.999998, 0.999999, 1.0,      1.0,      1.0},
         .activation = Options::SIGMOID,
         .rows = 4,
         .cols = 4,
         .rows_new = 5,
         .cols_new = 6,
         .channels = 1,
         .max_abs_diff = 1e-6},
        {.test_name = "SoftmaxActivationWithNoOutputResize",
         .inputs = {1.0,  2.0,  4.0,  2.0,  3.0,  5.0,  6.0,  1.5,
                    7.0,  10.0, 11.0, 4.0,  12.0, 15.0, 16.0, 18.5,
                    19.0, 20.0, 22.0, 23.0, 24.5, 23.4, 25.6, 28.3,
                    29.2, 30.0, 24.6, 29.2, 30.0, 24.9, 31.2, 30.3},
         .expected_outputs = {0.731059, 0.119203, 0.880797, 0.0109869, 0.952574,
                              0.000911051, 0.952574, 0.924142, 0.731059,
                              0.731059, 0.24974, 0.937027, 0.689974, 0.990048,
                              0.0060598, 0.28905},
         .activation = Options::SOFTMAX,
         .rows = 4,
         .cols = 4,
         .rows_new = 4,
         .cols_new = 4,
         .channels = 2,
         .max_abs_diff = 1e-6},
        {.test_name = "SoftmaxActivationWithOutputResize",
         .inputs = {1.0,  2.0,  4.0,  2.0,  3.0,  5.0,  6.0,  1.5,
                    7.0,  10.0, 11.0, 4.0,  12.0, 15.0, 16.0, 18.5,
                    19.0, 20.0, 22.0, 23.0, 24.5, 23.4, 25.6, 28.3,
                    29.2, 30.0, 24.6, 29.2, 30.0, 24.9, 31.2, 30.3},
         .expected_outputs = {0.731059,  0.425131, 0.246135, 0.753865, 0.445892,
                              0.0109869, 0.886119, 0.461259, 0.185506, 0.781934,
                              0.790618,  0.650195, 0.841816, 0.603901, 0.40518,
                              0.561962,  0.765871, 0.930584, 0.718733, 0.763744,
                              0.703402,  0.281989, 0.459635, 0.742634, 0.689974,
                              0.840011,  0.82605,  0.170058, 0.147555, 0.28905},
         .activation = Options::SOFTMAX,
         .rows = 4,
         .cols = 4,
         .rows_new = 5,
         .cols_new = 6,
         .channels = 2,
         .max_abs_diff = 1e-6},
    }),
    [](const testing::TestParamInfo<
        TensorsToSegmentationCalculatorTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace mediapipe
