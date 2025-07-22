// Copyright 2025 The MediaPipe Authors.
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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/test_util.h"
#include "third_party/OpenCV/core/hal/interface.h"
#include "third_party/OpenCV/core/mat.hpp"

namespace mediapipe {

using ::mediapipe::formats::MatView;
using ::testing::TestWithParam;

constexpr float kMaxColorDifference = 1.0;
constexpr float kMaxAlphaDifference = 1.0;
constexpr float kMaxAvgDifference = 0.1;

constexpr char kTestImagePath[] =
    "/mediapipe/calculators/image/testdata/googlelogo.png";
constexpr char kDarkGoldenImagePath[] =
    "/mediapipe/calculators/image/testdata/"
    "googlelogo_maskoverlay_0.png";
constexpr char kLightGoldenImagePath[] =
    "/mediapipe/calculators/image/testdata/"
    "googlelogo_maskoverlay_255.png";
constexpr char kTransparentGoldenImagePath[] =
    "/mediapipe/calculators/image/testdata/"
    "googlelogo_maskoverlay_alpha_0.png";

struct ConstantMaskParam {
  std::string test_name;
  float mask_value = 1;
  int image_rgb_value = 0;
  int channel_id = 1;
  std::string golden_image_path = "";
  std::optional<uint8_t> alpha_override = std::nullopt;
};

struct RegularMaskParam {
  std::string test_name;
  int width = 4;
  int height = 4;
  int image0_rgb_value = 100;
  int image1_rgb_value = 200;
  std::vector<float> mask_data;
  std::vector<int> golden_data;
};

using MaskOverlayCalculatorTest = TestWithParam<ConstantMaskParam>;
using MaskOverlayCalculatorTestRegularMask = TestWithParam<RegularMaskParam>;

// Create an ImageFrame with a constant pixel value.
ImageFrame CreateConstantImageFrame(
    int width, int height, int channels, uint8_t value,
    std::optional<uint8_t> alpha_override = std::nullopt) {
  auto image_format = (channels == 4) ? ImageFormat::SRGBA : ImageFormat::SRGB;
  ImageFrame frame(image_format, width, height, /*alignment_boundary =*/1);

  uint8_t* pixel_data = frame.MutablePixelData();
  if (channels == 4 && alpha_override.has_value()) {
    for (int i = 0; i < width * height; ++i) {
      pixel_data[i * 4 + 0] = value;                   // R
      pixel_data[i * 4 + 1] = value;                   // G
      pixel_data[i * 4 + 2] = value;                   // B
      pixel_data[i * 4 + 3] = alpha_override.value();  // A
    }
  } else {
    for (int i = 0; i < width * height * channels; ++i) {
      pixel_data[i] = value;
    }
  }
  return frame;
}

CalculatorGraphConfig GetMaskGraphConfig(int mask_channel_id) {
  return ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
      R"pb(
        input_stream: "input_video0_cpu"
        input_stream: "input_video1_cpu"
        input_stream: "mask_cpu"
        output_stream: "output_cpu"
        node {
          calculator: "ImageFrameToGpuBufferCalculator"
          input_stream: "input_video0_cpu"
          output_stream: "input_video0_gpu"
        }
        node {
          calculator: "ImageFrameToGpuBufferCalculator"
          input_stream: "input_video1_cpu"
          output_stream: "input_video1_gpu"
        }
        node {
          calculator: "ImageFrameToGpuBufferCalculator"
          input_stream: "mask_cpu"
          output_stream: "mask_gpu"
        }
        node {
          calculator: "MaskOverlayCalculator"
          input_stream: "VIDEO:0:input_video0_gpu"
          input_stream: "VIDEO:1:input_video1_gpu"
          input_stream: "MASK:mask_gpu"
          output_stream: "OUTPUT:output_gpu"
          options {
            [mediapipe.MaskOverlayCalculatorOptions.ext] { mask_channel: $0 }
          }
        }
        node {
          calculator: "GpuBufferToImageFrameCalculator"
          input_stream: "output_gpu"
          output_stream: "output_cpu"
        }
      )pb",
      mask_channel_id));
}

CalculatorGraphConfig GetConstantMaskGraphConfig(int mask_channel_id) {
  return ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
      R"pb(
        input_stream: "input_video0_cpu"
        input_stream: "input_video1_cpu"
        input_stream: "const_mask"
        output_stream: "output_cpu"
        node {
          calculator: "ImageFrameToGpuBufferCalculator"
          input_stream: "input_video0_cpu"
          output_stream: "input_video0_gpu"
        }
        node {
          calculator: "ImageFrameToGpuBufferCalculator"
          input_stream: "input_video1_cpu"
          output_stream: "input_video1_gpu"
        }
        node {
          calculator: "MaskOverlayCalculator"
          input_stream: "VIDEO:0:input_video0_gpu"
          input_stream: "VIDEO:1:input_video1_gpu"
          input_stream: "CONST_MASK:const_mask"
          output_stream: "OUTPUT:output_gpu"
          options {
            [mediapipe.MaskOverlayCalculatorOptions.ext] { mask_channel: $0 }
          }
        }
        node {
          calculator: "GpuBufferToImageFrameCalculator"
          input_stream: "output_gpu"
          output_stream: "output_cpu"
        }
      )pb",
      mask_channel_id));
}

TEST_P(MaskOverlayCalculatorTest, TestConstantMask) {
  const ConstantMaskParam& test_case = GetParam();
  CalculatorGraphConfig graph_config =
      GetConstantMaskGraphConfig(test_case.channel_id);

  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_cpu", &graph_config, &output_packets);

  int channels = test_case.alpha_override.has_value() ? 4 : 3;
  // Set packet for the first image frame.
  std::string input_image_path =
      file::JoinPath(GetTestRootDir(), kTestImagePath);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageFrame> input_frame0,
                          LoadTestImage(input_image_path, ImageFormat::SRGBA));
  int frame_width = input_frame0->Width();
  int frame_height = input_frame0->Height();
  auto input_packet0 = MakePacket<ImageFrame>(std::move(*input_frame0));

  // Set packet for the second image frame.
  auto input_frame1 = CreateConstantImageFrame(
      frame_width, frame_height, channels, test_case.image_rgb_value,
      test_case.alpha_override);
  auto input_packet1 = MakePacket<ImageFrame>(std::move(input_frame1));

  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream("input_video0_cpu",
                                            input_packet0.At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream("input_video1_cpu",
                                            input_packet1.At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "const_mask", MakePacket<float>(test_case.mask_value).At(Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_packets.size(), 1);

  const auto& output_frame = output_packets[0].Get<ImageFrame>();

  // Load the golden image and compare it to the output frame.
  std::string golden_image_path_dir =
      file::JoinPath(GetTestRootDir(), test_case.golden_image_path);
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ImageFrame> golden_frame,
      LoadTestImage(golden_image_path_dir, ImageFormat::SRGBA));
  auto diff_image = std::make_unique<ImageFrame>(
      golden_frame->Format(), golden_frame->Width(), golden_frame->Height());

  absl::Status comparison =
      CompareImageFrames(*golden_frame, output_frame, kMaxColorDifference,
                         kMaxAlphaDifference, kMaxAvgDifference, diff_image);
  MP_EXPECT_OK(comparison);

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

INSTANTIATE_TEST_SUITE_P(
    VerifyOutputMaxCount, MaskOverlayCalculatorTest,
    testing::ValuesIn<ConstantMaskParam>({
        {.test_name = "RGBChannelDark",
         .mask_value = 0.5,
         .image_rgb_value = 0,
         .channel_id = 1,
         .golden_image_path = kDarkGoldenImagePath},
        {.test_name = "RGBChannelLight",
         .mask_value = 0.5,
         .image_rgb_value = 255,
         .channel_id = 1,
         .golden_image_path = kLightGoldenImagePath},
        {.test_name = "AlphaChannelTransparent",
         .mask_value = 0.5,
         .image_rgb_value = 0,
         .channel_id = 2,
         .golden_image_path = kTransparentGoldenImagePath,
         .alpha_override = 0},
    }),
    [](const testing::TestParamInfo<ConstantMaskParam>& info) {
      return info.param.test_name;
    });

TEST_P(MaskOverlayCalculatorTestRegularMask, TestRegularMask) {
  const auto& [test_name, input_width, input_height, image0_rgb_value,
               image1_rgb_value, mask_data, golden_data] = GetParam();
  CalculatorGraphConfig graph_config = GetMaskGraphConfig(1);

  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_cpu", &graph_config, &output_packets);

  // set packet for the first image frame.
  auto input_frame0 = CreateConstantImageFrame(input_width, input_height, 3,
                                               image0_rgb_value);  // SRGB
  auto input_packet0 = MakePacket<ImageFrame>(std::move(input_frame0));

  // set packet for the second image frame.
  auto input_frame1 = CreateConstantImageFrame(input_width, input_height, 3,
                                               image1_rgb_value);  // SRGB
  auto input_packet1 = MakePacket<ImageFrame>(std::move(input_frame1));

  cv::Mat mask_source_cv(input_height, input_width, CV_32FC1,
                         const_cast<float*>(mask_data.data()));
  ImageFrame mask_image(ImageFormat::VEC32F1, input_width, input_height);
  mask_source_cv.copyTo(formats::MatView(&mask_image));
  auto mask_packet = MakePacket<ImageFrame>(std::move(mask_image));

  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream("input_video0_cpu",
                                            input_packet0.At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream("input_video1_cpu",
                                            input_packet1.At(Timestamp(0))));
  MP_ASSERT_OK(
      graph.AddPacketToInputStream("mask_cpu", mask_packet.At(Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(output_packets.size(), 1);

  const auto& output_frame = output_packets[0].Get<ImageFrame>();
  cv::Mat result_mat = MatView(&output_frame);

  for (int i = 0; i < input_height; ++i) {
    for (int j = 0; j < input_width; ++j) {
      int val = static_cast<int>(golden_data[i * input_width + j]);
      uchar* pixel_ptr = result_mat.ptr<uchar>(i, j);
      EXPECT_EQ(val, (int)pixel_ptr[0]);
    }
  }

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

INSTANTIATE_TEST_SUITE_P(
    TestRegularMask, MaskOverlayCalculatorTestRegularMask,
    testing::ValuesIn<RegularMaskParam>({
        {.test_name = "RegularMask0",
         .width = 4,
         .height = 4,
         .image0_rgb_value = 100,
         .image1_rgb_value = 200,
         .mask_data =
             {
                 0.00,
                 0.00,
                 0.00,
                 0.00,  //
                 0.00,
                 1.00,
                 1.00,
                 0.00,  //
                 0.00,
                 1.00,
                 1.00,
                 0.00,  //
                 0.00,
                 0.00,
                 0.00,
                 0.00,  //
             },
         .golden_data =
             {
                 100,
                 100,
                 100,
                 100,  //
                 100,
                 200,
                 200,
                 100,  //
                 100,
                 200,
                 200,
                 100,  //
                 100,
                 100,
                 100,
                 100,  //
             }},
        {.test_name = "RegularMask1",
         .width = 4,
         .height = 4,
         .image0_rgb_value = 100,
         .image1_rgb_value = 200,
         .mask_data =
             {
                 0.50,
                 0.00,
                 0.00,
                 0.00,  //
                 0.00,
                 0.50,
                 1.00,
                 0.00,  //
                 0.00,
                 1.00,
                 1.00,
                 0.00,  //
                 0.00,
                 0.00,
                 0.00,
                 0.00,  //
             },
         .golden_data =
             {
                 150,
                 100,
                 100,
                 100,  //
                 100,
                 150,
                 200,
                 100,  //
                 100,
                 200,
                 200,
                 100,  //
                 100,
                 100,
                 100,
                 100,  //
             }},
    }),
    [](const testing::TestParamInfo<RegularMaskParam>& info) {
      return info.param.test_name;
    });

TEST(MaskOverlayCalculatorTest, NoMaskInputFails) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MaskOverlayCalculator"
        input_stream: "VIDEO:0:input_video"
        input_stream: "VIDEO:1:input_video1"
        output_stream: "OUTPUT:output"
      )pb");

  CalculatorRunner runner(node_config);

  ASSERT_THAT(
      runner.Run(),
      StatusIs(absl::StatusCode::kNotFound,
               testing::HasSubstr("mask input stream must be present")));
}

}  // namespace mediapipe
