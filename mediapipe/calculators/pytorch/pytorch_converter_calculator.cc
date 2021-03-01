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
#endif  // MEDIAPIPE_IOS

namespace mediapipe {

namespace {
::mediapipe::Status EnsureFormat(const ImageFrame& image_frame) {
  const ImageFormat::Format format = image_frame.Format();
  if (!(format == ImageFormat::SRGB)) {
    RET_CHECK_FAIL() << "Unsupported input format.";
  }
  return ::mediapipe::OkStatus();
}

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageGpuTag[] = "IMAGE_GPU";
constexpr char kTensorsTag[] = "TENSORS";

using Output = torch::jit::IValue;
using Outputs = std::vector<Output>;
}  // namespace

// Calculator for normalizing and converting an ImageFrame
// into a PyTorchTensor.
//
// This calculator is designed to be used with the PyTorchInferenceCalculator,
// as a pre-processing step for calculator inputs.
//
// IMAGE and IMAGE_GPU inputs are normalized to [0;1].
//
// Input:
//  One of the following tags:
//  IMAGE - ImageFrame.
//  IMAGE_GPU - GpuBuffer.
//
// Output:
//  One of the following tags:
//  TENSORS - Vector of torch::jit::IValue residing on CPU.
//
// Example use:
// node {
//   calculator: "PyTorchConverterCalculator"
//   input_stream: "IMAGE:input_image"
//   output_stream: "TENSORS:image_tensor"
//   options: {
//     [mediapipe.PyTorchConverterCalculatorOptions.ext] {
//       per_channel_normalizations: {sub:0.485 div:0.229}
//       per_channel_normalizations: {sub:0.456 div:0.224}
//       per_channel_normalizations: {sub:0.406 div:0.225}
//     }
//   }
// }
//
// IMPORTANT Notes:
//  If given an IMAGE_GPU, PyTorch will convert TENSORS to CPU.
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
  bool has_tensors_tag_;
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

  if (has_image_tag) {
    cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
  }
  if (has_image_gpu_tag) {
#if defined(MEDIAPIPE_IOS)
    cc->Inputs().Tag(kImageGpuTag).Set<GpuBuffer>();
#else
    RET_CHECK_FAIL() << "GPU processing not enabled.";
#endif
  }

  cc->Outputs().Tag(kTensorsTag).Set<Outputs>();

  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PyTorchConverterCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<::mediapipe::PyTorchConverterCalculatorOptions>();

  has_image_tag_ = cc->Inputs().HasTag(kImageTag);
  has_image_gpu_tag_ = cc->Inputs().HasTag(kImageGpuTag);
  has_tensors_tag_ = cc->Outputs().HasTag(kTensorsTag);

  if (has_image_gpu_tag_) {
#if !defined(MEDIAPIPE_IOS)
    RET_CHECK_FAIL() << "GPU processing not enabled.";
#endif  // MEDIAPIPE_IOS
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PyTorchConverterCalculator::Process(CalculatorContext* cc) {
  cv::Mat image;
  // Acquire input packet as ImageFrame image, if packet is not empty,
  if (has_image_gpu_tag_) {
#if defined(MEDIAPIPE_IOS) && MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    if (cc->Inputs().Tag(kImageGpuTag).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }
    const auto& input = cc->Inputs().Tag(kImageGpuTag).Get<GpuBuffer>();
    std::unique_ptr<ImageFrame> frame =
        CreateImageFrameForCVPixelBuffer(input.GetCVPixelBufferRef());
    ImageFrame image_frame = frame.release();
    MP_RETURN_IF_ERROR(EnsureFormat(image_frame));
    image = mediapipe::formats::MatView(image_frame);
#else
    RET_CHECK_FAIL() << "GPU processing is not enabled.";
#endif
  }
  if (has_image_tag_) {
    if (cc->Inputs().Tag(kImageTag).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }
    const ImageFrame& image_frame =
        cc->Inputs().Tag(kImageTag).Get<ImageFrame>();
    MP_RETURN_IF_ERROR(EnsureFormat(image_frame));
    image = mediapipe::formats::MatView(&image_frame);
  }

  const int num_channels = image.channels();
  RET_CHECK_EQ(num_channels, 3) << "Only RGB images are supported";
  const int width = image.cols;
  const int height = image.rows;
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  cv::Mat img_float;
  // Normalize to [0;1]
  image.convertTo(img_float, CV_32F, 1.0 / 255);
  auto img_tensor = torch::from_blob(img_float.data, {1, width, height, 3});
  // Permute from NWHC to NCWH
  img_tensor = img_tensor.permute({0, 3, 1, 2});

  if (options_.per_channel_normalizations().size() > 0) {
    // Further normalize each channel of input image
    for (int i = 0; i < num_channels; ++i) {
      const auto& subdiv = options_.per_channel_normalizations(i);
      const float sub = subdiv.sub();
      const float div = subdiv.div();
      img_tensor[0][i] = img_tensor[0][i].sub_(sub).div_(div);
    }
  }

  if (has_tensors_tag_) {
    auto output_tensors =
        absl::make_unique<Outputs, std::initializer_list<Output>>({
            img_tensor.cpu(),
        });
    cc->Outputs()
        .Tag(kTensorsTag)
        .Add(output_tensors.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  } else {
    RET_CHECK_FAIL() << "Unsupported output kind.";
  }
}

::mediapipe::Status PyTorchConverterCalculator::Close(CalculatorContext* cc) {
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
