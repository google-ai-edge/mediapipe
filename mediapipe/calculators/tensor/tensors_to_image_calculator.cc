// Copyright 2021 The MediaPipe Authors.
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

#include <vector>
#include <iostream>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/util/resource_util.h"
#include "tensorflow/lite/interpreter.h"

namespace
{
  constexpr char kTensorsTag[] = "TENSORS";
  constexpr char kOutputSizeTag[] = "OUTPUT_SIZE";
  constexpr char kImageTag[] = "IMAGE";

  absl::StatusOr<std::tuple<int, int, int>> GetHwcFromDims(
      const std::vector<int> &dims)
  {
    if (dims.size() == 3)
    {
      return std::make_tuple(dims[0], dims[1], dims[2]);
    }
    else if (dims.size() == 4)
    {
      // BHWC format check B == 1
      RET_CHECK_EQ(1, dims[0]) << "Expected batch to be 1 for BHWC heatmap";
      return std::make_tuple(dims[1], dims[2], dims[3]);
    }
    else
    {
      RET_CHECK(false) << "Invalid shape for segmentation tensor " << dims.size();
    }
  }
} // namespace

namespace mediapipe
{

  // Converts Tensors from a tflite segmentation model to an image.
  //
  // Performs optional upscale to OUTPUT_SIZE dimensions if provided,
  // otherwise the image is the same size as input tensor.
  //
  //
  //
  // Inputs:
  //   One of the following TENSORS tags:
  //   TENSORS: Vector of Tensor,
  //            The tensor dimensions are specified in this calculator's options.
  //   OUTPUT_SIZE(optional): std::pair<int, int>,
  //                          If provided, the size to upscale mask to.
  //
  // Output:
  //   IMAGE: An Image output, RGBA.
  //
  //
  // Usage example:
  // node {
  //   calculator: "TensorsToImageCalculator"
  //   input_stream: "TENSORS:tensors"
  //   input_stream: "OUTPUT_SIZE:size"
  //   output_stream: "IMAGE:image"
  // }
  //
  // TODO Refactor and add support for other backends/platforms.
  //
  class TensorsToImageCalculator : public CalculatorBase
  {
  public:
    static absl::Status GetContract(CalculatorContract *cc);

    absl::Status Open(CalculatorContext *cc) override;
    absl::Status Process(CalculatorContext *cc) override;
    absl::Status Close(CalculatorContext *cc) override;

  private:
    absl::Status ProcessCpu(CalculatorContext *cc);
    
 };
  REGISTER_CALCULATOR(TensorsToImageCalculator);

  // static
  absl::Status TensorsToImageCalculator::GetContract(
      CalculatorContract *cc)
  {
    RET_CHECK(!cc->Inputs().GetTags().empty());
    RET_CHECK(!cc->Outputs().GetTags().empty());

    // Inputs.
    cc->Inputs().Tag(kTensorsTag).Set<std::vector<Tensor>>();
    if (cc->Inputs().HasTag(kOutputSizeTag))
    {
      cc->Inputs().Tag(kOutputSizeTag).Set<std::pair<int, int>>();
    }

    // Outputs.
    cc->Outputs().Tag(kImageTag).Set<Image>();

    return absl::OkStatus();
  }

  absl::Status TensorsToImageCalculator::Open(CalculatorContext *cc)
  {
    cc->SetOffset(TimestampDiff(0));

    return absl::OkStatus();
  }

  absl::Status TensorsToImageCalculator::Process(CalculatorContext *cc)
  {
    if (cc->Inputs().Tag(kTensorsTag).IsEmpty())
    {
      return absl::OkStatus();
    }

    const auto &input_tensors =
        cc->Inputs().Tag(kTensorsTag).Get<std::vector<Tensor>>();

    MP_RETURN_IF_ERROR(ProcessCpu(cc));

    return absl::OkStatus();
  }

  absl::Status TensorsToImageCalculator::Close(CalculatorContext *cc)
  {

    return absl::OkStatus();
  }

  absl::Status TensorsToImageCalculator::ProcessCpu(
      CalculatorContext *cc)
  {
    // Get input streams, and dimensions.
    const auto &input_tensors =
        cc->Inputs().Tag(kTensorsTag).Get<std::vector<Tensor>>();
    ASSIGN_OR_RETURN(auto hwc, GetHwcFromDims(input_tensors[0].shape().dims));
    auto [tensor_height, tensor_width, tensor_channels] = hwc;
    int output_width = tensor_width, output_height = tensor_height;
    if (cc->Inputs().HasTag(kOutputSizeTag))
    {
      const auto &size =
          cc->Inputs().Tag(kOutputSizeTag).Get<std::pair<int, int>>();
      output_width = size.first;
      output_height = size.second;
    }

    cv::Mat image_mat(cv::Size(tensor_width, tensor_height), CV_32FC1);

    // Wrap input tensor.
    auto raw_input_tensor = &input_tensors[0];
    auto raw_input_view = raw_input_tensor->GetCpuReadView();
    const float *raw_input_data = raw_input_view.buffer<float>();
    cv::Mat tensor_mat(cv::Size(tensor_width, tensor_height),
                       CV_MAKETYPE(CV_32F, tensor_channels),
                       const_cast<float *>(raw_input_data));

    std::vector<cv::Mat> channels(4);
    cv::split(tensor_mat, channels);
    for (auto ch : channels)
      ch = (ch + 1) * 127.5;

    cv::merge(channels, tensor_mat);

    cv::convertScaleAbs(tensor_mat, tensor_mat);

    // Send out image as CPU packet.
    std::shared_ptr<ImageFrame> image_frame = std::make_shared<ImageFrame>(
        ImageFormat::SRGB, output_width, output_height);
    std::unique_ptr<Image> output_image = absl::make_unique<Image>(image_frame);
    auto output_mat = formats::MatView(output_image.get());
    // Upsample image into output.
    cv::resize(tensor_mat, *output_mat,
               cv::Size(output_width, output_height));
    cc->Outputs().Tag(kImageTag).Add(output_image.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }

} // namespace mediapipe
