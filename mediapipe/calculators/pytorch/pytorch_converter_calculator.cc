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

#include <string>
#include <vector>

#include "mediapipe/calculators/pytorch/pytorch_converter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "torch/script.h"
#include "torch/torch.h"

#if defined(MEDIAPIPE_IOS)
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/objc/util.h"
#endif  // iOS

namespace mediapipe {

namespace {
constexpr char kImageTag[] = "IMAGE";
constexpr char kImageGpuTag[] = "IMAGE_GPU";
constexpr char kTensorsTag[] = "TENSORS";

using Outputs = std::vector<torch::jit::IValue>;
}  // namespace

// Calculator for normalizing and converting an ImageFrame
// into a PyTorchTensor.
//
// This calculator is designed to be used with the PyTorchInferenceCalculator,
// as a pre-processing step for calculator inputs.
//
// IMAGE inputs are normalized to [-1,1] (default) or [0,1],
// specified by options (unless outputting a quantized tensor).
//
// Input:
//  One of the following tags:
//  IMAGE - ImageFrame (assumed to be 8-bit or 32-bit data).
//
// Output:
//  One of the following tags:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32, or kTfLiteUint8.
//
// Example use:
// node {
//   calculator: "PyTorchConverterCalculator"
//   input_stream: "IMAGE:input_image"
//   output_stream: "TENSORS:image_tensor"
//   options: {
//     [mediapipe.PyTorchConverterCalculatorOptions.ext] {
//       zero_center: true
//     }
//   }
// }
//
// IMPORTANT Notes:
//  This calculator uses FixedSizeInputStreamHandler by default.
//
class PyTorchConverterCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::PyTorchConverterCalculatorOptions options_;
  bool has_image_tag_;
  bool has_image_gpu_tag_;
};
REGISTER_CALCULATOR(PyTorchConverterCalculator);

::mediapipe::Status PyTorchConverterCalculator::GetContract(
    CalculatorContract* cc) {
  const bool has_image_tag = cc->Inputs().HasTag(kImageTag);
  const bool has_image_gpu_tag = cc->Inputs().HasTag(kImageGpuTag);
  // Confirm only one of the input streams is present.
  RET_CHECK(has_image_tag ^ has_image_gpu_tag);

  const bool has_tensors_tag = cc->Outputs().HasTag(kTensorsTag);
  RET_CHECK(has_tensors_tag);

  if (has_image_tag) cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
  if (has_image_gpu_tag) {
#if defined(MEDIAPIPE_IOS)
    cc->Inputs().Tag(kImageGpuTag).Set<GpuBuffer>();
#else
    RET_CHECK_FAIL() << "GPU processing not enabled.";
#endif
  }

  if (has_tensors_tag) cc->Outputs().Tag(kTensorsTag).Set<Outputs>();

  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PyTorchConverterCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<::mediapipe::PyTorchConverterCalculatorOptions>();

  has_image_tag_ = cc->Inputs().HasTag(kImageTag);
  has_image_gpu_tag_ = cc->Inputs().HasTag(kImageGpuTag);

  if (has_image_gpu_tag_) {
#if !defined(MEDIAPIPE_IOS)
    RET_CHECK_FAIL() << "GPU processing not enabled.";
#endif
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PyTorchConverterCalculator::Process(CalculatorContext* cc) {
  cv::Mat image;
  if (has_image_gpu_tag_) {
#if defined(MEDIAPIPE_IOS) && MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    const auto& input = cc->Inputs().Tag(kImageGpuTag).Get<GpuBuffer>();
    std::unique_ptr<ImageFrame> frame =
        CreateImageFrameForCVPixelBuffer(input.GetCVPixelBufferRef());
    image = mediapipe::formats::MatView(frame.release());
#else
    RET_CHECK_FAIL() << "GPU processing is not enabled.";
#endif
  }
  if (has_image_tag_) {
    auto& output_frame =
        cc->Inputs().Tag(kImageTag).Get<mediapipe::ImageFrame>();
    image = mediapipe::formats::MatView(&output_frame);
  }
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  const auto width = image.cols, height = image.rows,
             num_channels = image.channels();
  RET_CHECK_EQ(num_channels, 3) << "Only RGB images are supported for now";

  cv::Mat img_float;
  image.convertTo(img_float, CV_32F, 1.0 / 255);
  auto img_tensor = torch::from_blob(img_float.data, {1, width, height, 3});
  img_tensor = img_tensor.permute({0, 3, 1, 2});  // To NCWH
  if (options_.per_channel_normalizations().size() > 0)
    for (int i = 0; i < num_channels; ++i) {
      const auto& subdiv = options_.per_channel_normalizations(i);
      img_tensor[0][i] = img_tensor[0][i].sub_(subdiv.sub()).div_(subdiv.div());
    }

  auto output_tensors = absl::make_unique<Outputs>();
  output_tensors->reserve(1);
  output_tensors->emplace_back(img_tensor.cpu());
  cc->Outputs()
      .Tag(kTensorsTag)
      .Add(output_tensors.release(), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PyTorchConverterCalculator::Close(CalculatorContext* cc) {
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
