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

#include <iostream>

#include "mediapipe/calculators/tensorflow/tensor_to_image_frame_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace mediapipe {

namespace tf = ::tensorflow;
namespace {

constexpr char kImage[] = "IMAGE";
constexpr char kTensor[] = "TENSOR";

}  // namespace

// Input:
//  Tensor of type DT_FLOAT, with values between 0-255 (SRGB or GRAY8). The
//  shape can be HxWx{3,1} or simply HxW.
//
//  Optionally supports a scale factor that can scale 0-1 value ranges to 0-255.
//
// Output:
//  ImageFrame containing the values of the tensor cast as uint8 (SRGB or GRAY8)
//
// Possible extensions: support other input ranges, maybe 4D tensors.
class TensorToImageFrameCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  float scale_factor_;
};

REGISTER_CALCULATOR(TensorToImageFrameCalculator);

::mediapipe::Status TensorToImageFrameCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one input stream is supported.";
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "One input stream must be provided.";
  RET_CHECK(cc->Inputs().HasTag(kTensor))
      << "An input stream for tag: " << kTensor << " must be provided.";
  cc->Inputs().Tag(kTensor).Set<tf::Tensor>(
      // Input Tensor.
  );
  cc->Outputs().Tag(kImage).Set<ImageFrame>(
      // Output ImageFrame.
  );
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TensorToImageFrameCalculator::Open(CalculatorContext* cc) {
  scale_factor_ =
      cc->Options<TensorToImageFrameCalculatorOptions>().scale_factor();
  cc->SetOffset(TimestampDiff(0));
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TensorToImageFrameCalculator::Process(
    CalculatorContext* cc) {
  const tf::Tensor& input_tensor = cc->Inputs().Tag(kTensor).Get<tf::Tensor>();
  int32 depth = 1;
  if (input_tensor.dims() != 2) {  // Depth is 1 for 2D tensors.
    CHECK(3 == input_tensor.dims())
        << "Only 2 or 3-D Tensors can be converted to frames. Instead got: "
        << input_tensor.dims();
    depth = input_tensor.dim_size(2);
    if (depth != 1) {
      RET_CHECK_EQ(depth, 3) << "Output tensor depth must be 3 or 1.";
    }
  }
  const int32 total_size =
      input_tensor.dim_size(0) * input_tensor.dim_size(1) * depth;
  std::unique_ptr<uint8[]> buffer(new uint8[total_size]);
  auto data = input_tensor.flat<float>().data();
  for (int i = 0; i < total_size; ++i) {
    float d = scale_factor_ * data[i];
    if (d < 0) d = 0;
    if (d > 255) d = 255;
    buffer[i] = d;
  }

  ::std::unique_ptr<ImageFrame> output;
  if (depth == 3) {
    output = ::absl::make_unique<ImageFrame>(
        ImageFormat::SRGB, input_tensor.dim_size(1), input_tensor.dim_size(0),
        input_tensor.dim_size(1) * 3, buffer.release());
  } else if (depth == 1) {
    output = ::absl::make_unique<ImageFrame>(
        ImageFormat::GRAY8, input_tensor.dim_size(1), input_tensor.dim_size(0),
        input_tensor.dim_size(1), buffer.release());
  } else {
    return ::mediapipe::InvalidArgumentError("Unrecognized image depth.");
  }
  cc->Outputs().Tag(kImage).Add(output.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
