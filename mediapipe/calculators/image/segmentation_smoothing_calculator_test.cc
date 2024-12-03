// Copyright 2018 The MediaPipe Authors.
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

#include "absl/log/absl_log.h"
#include "mediapipe/calculators/image/segmentation_smoothing_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

// 4x4 VEC32F1, center 2x2 block set at ~250
const float mask_data[] = {
    0.00, 0.00, 0.00, 0.00,  //
    0.00, 0.98, 0.98, 0.00,  //
    0.00, 0.98, 0.98, 0.00,  //
    0.00, 0.00, 0.00, 0.00,  //
};

void RunGraph(Packet curr_packet, Packet prev_packet, bool use_gpu, float ratio,
              cv::Mat* result) {
  CalculatorGraphConfig graph_config;
  if (use_gpu) {
    graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
        R"pb(
          input_stream: "curr_mask"
          input_stream: "prev_mask"
          output_stream: "new_mask"
          node {
            calculator: "ImageCloneCalculator"
            input_stream: "curr_mask"
            output_stream: "curr_mask_gpu"
            options: {
              [mediapipe.ImageCloneCalculatorOptions.ext] {
                output_on_gpu: true
              }
            }
          }
          node {
            calculator: "ImageCloneCalculator"
            input_stream: "prev_mask"
            output_stream: "prev_mask_gpu"
            options: {
              [mediapipe.ImageCloneCalculatorOptions.ext] {
                output_on_gpu: true
              }
            }
          }
          node {
            calculator: "SegmentationSmoothingCalculator"
            input_stream: "MASK:curr_mask_gpu"
            input_stream: "MASK_PREVIOUS:prev_mask_gpu"
            output_stream: "MASK_SMOOTHED:new_mask"
            node_options {
              [type.googleapis.com/
               mediapipe.SegmentationSmoothingCalculatorOptions]: {
                combine_with_previous_ratio: $0
              }
            }
          }
        )pb",
        ratio));
  } else {
    graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
        R"pb(
          input_stream: "curr_mask"
          input_stream: "prev_mask"
          output_stream: "new_mask"
          node {
            calculator: "SegmentationSmoothingCalculator"
            input_stream: "MASK:curr_mask"
            input_stream: "MASK_PREVIOUS:prev_mask"
            output_stream: "MASK_SMOOTHED:new_mask"
            node_options {
              [type.googleapis.com/
               mediapipe.SegmentationSmoothingCalculatorOptions]: {
                combine_with_previous_ratio: $0
              }
            }
          }
        )pb",
        ratio));
  }
  std::vector<Packet> output_packets;
  tool::AddVectorSink("new_mask", &graph_config, &output_packets);
  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(
      graph.AddPacketToInputStream("curr_mask", curr_packet.At(Timestamp(0))));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("prev_mask", prev_packet.At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(1, output_packets.size());

  Image result_image = output_packets[0].Get<Image>();
  auto result_mat = formats::MatView(&result_image);
  result_mat->copyTo(*result);

  // Fully close graph at end, otherwise calculator+Images are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("curr_mask"));
  MP_ASSERT_OK(graph.CloseInputStream("prev_mask"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

void RunTest(bool use_gpu, float mix_ratio, cv::Mat& test_result) {
  cv::Mat mask_mat(cv::Size(4, 4), CV_32FC1, const_cast<float*>(mask_data));
  cv::Mat curr_mat = mask_mat;
  // 3x3 blur of 250 block produces all pixels '111'.
  cv::Mat prev_mat;
  cv::blur(mask_mat, prev_mat, cv::Size(3, 3));

  Packet curr_packet = MakePacket<Image>(std::make_unique<ImageFrame>(
      ImageFormat::VEC32F1, curr_mat.size().width, curr_mat.size().height));
  curr_mat.copyTo(*formats::MatView(&(curr_packet.Get<Image>())));
  Packet prev_packet = MakePacket<Image>(std::make_unique<ImageFrame>(
      ImageFormat::VEC32F1, prev_mat.size().width, prev_mat.size().height));
  prev_mat.copyTo(*formats::MatView(&(prev_packet.Get<Image>())));

  cv::Mat result;
  RunGraph(curr_packet, prev_packet, use_gpu, mix_ratio, &result);

  ASSERT_EQ(curr_mat.rows, result.rows);
  ASSERT_EQ(curr_mat.cols, result.cols);
  ASSERT_EQ(curr_mat.type(), result.type());
  result.copyTo(test_result);

  if (mix_ratio == 1.0) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        float in = curr_mat.at<float>(i, j);
        float out = result.at<float>(i, j);
        // Since the input has high value (250), it has low uncertainty.
        // So the output should have changed lower (towards prev),
        // but not too much.
        if (in > 0) EXPECT_NE(in, out);
        EXPECT_NEAR(in, out, 3.0 / 255.0);
      }
    }
  } else if (mix_ratio == 0.0) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        float in = curr_mat.at<float>(i, j);
        float out = result.at<float>(i, j);
        EXPECT_EQ(in, out);  // Output should match current.
      }
    }
  } else {
    ABSL_LOG(ERROR) << "invalid ratio";
  }
}

TEST(SegmentationSmoothingCalculatorTest, TestSmoothing) {
  bool use_gpu;
  float mix_ratio;

  use_gpu = false;
  mix_ratio = 0.0;
  cv::Mat cpu_0;
  RunTest(use_gpu, mix_ratio, cpu_0);

  use_gpu = false;
  mix_ratio = 1.0;
  cv::Mat cpu_1;
  RunTest(use_gpu, mix_ratio, cpu_1);

  use_gpu = true;
  mix_ratio = 1.0;
  cv::Mat gpu_1;
  RunTest(use_gpu, mix_ratio, gpu_1);

  // CPU & GPU should match.
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      float gpu = gpu_1.at<float>(i, j);
      float cpu = cpu_1.at<float>(i, j);
      EXPECT_EQ(cpu, gpu);
    }
  }
}

}  // namespace
}  // namespace mediapipe
