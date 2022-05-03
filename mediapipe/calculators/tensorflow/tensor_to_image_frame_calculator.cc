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
//  Tensor of type DT_FLOAT or DT_UINT8, with values between 0-255
//  (SRGB or GRAY8). The shape can be HxWx{3,1} or simply HxW.
//
//  For DT_FLOAT tensors, optionally supports a scale factor that can scale 0-1
//  value ranges to 0-255.
//
// Output:
//  ImageFrame containing the values of the tensor cast as uint8 (SRGB or GRAY8)
//
// Possible extensions: support other input ranges, maybe 4D tensors.
//
// Example:
// node {
//   calculator: "TensorToImageFrameCalculator"
//   input_stream: "TENSOR:3d_float_tensor"
//   output_stream: "IMAGE:image_frame"
//   options {
//     [mediapipe.TensorToImageFrameCalculatorOptions.ext] {
//       scale_factor: 1.0  # set to 255.0 for [0,1] -> [0,255] scaling
//     }
//   }
// }
class TensorToImageFrameCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  float scale_factor_;
};

REGISTER_CALCULATOR(TensorToImageFrameCalculator);

absl::Status TensorToImageFrameCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1)
      << "Only one output stream is supported.";
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
  return absl::OkStatus();
}

absl::Status TensorToImageFrameCalculator::Open(CalculatorContext* cc) {
  scale_factor_ =
      cc->Options<TensorToImageFrameCalculatorOptions>().scale_factor();
  cc->SetOffset(TimestampDiff(0));
  return absl::OkStatus();
}

absl::Status TensorToImageFrameCalculator::Process(CalculatorContext* cc) {
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
  int32 height = input_tensor.dim_size(0);
  int32 width = input_tensor.dim_size(1);
  auto format = (depth == 3 ? ImageFormat::SRGB : ImageFormat::GRAY8);
  const int32 total_size = height * width * depth;

  ::std::unique_ptr<const ImageFrame> output;
  if (input_tensor.dtype() == tensorflow::DT_FLOAT) {
    // Allocate buffer with alignments.
    std::unique_ptr<uint8_t[]> buffer(
        new (std::align_val_t(EIGEN_MAX_ALIGN_BYTES)) uint8_t[total_size]);
    auto data = input_tensor.flat<float>().data();
    for (int i = 0; i < total_size; ++i) {
      float d = scale_factor_ * data[i];
      if (d < 0) d = 0;
      if (d > 255) d = 255;
      buffer[i] = d;
    }
    output = ::absl::make_unique<ImageFrame>(
        format, width, height, width * depth, buffer.release(),
        [total_size](uint8* ptr) {
          ::operator delete[](ptr, total_size,
                              std::align_val_t(EIGEN_MAX_ALIGN_BYTES));
        });
  } else if (input_tensor.dtype() == tensorflow::DT_UINT8) {
    if (scale_factor_ != 1.0) {
      return absl::InvalidArgumentError("scale_factor_ given for uint8 tensor");
    }
    // tf::Tensor has internally ref-counted buffer. The following code make the
    // ImageFrame own the copied Tensor through the deleter, which increases
    // the refcount of the buffer and allow us to use the shared buffer as the
    // image. This allows us to create an ImageFrame object without copying
    // buffer. const ImageFrame prevents the buffer from being modified later.
    auto copy = new tf::Tensor(input_tensor);
    output = ::absl::make_unique<const ImageFrame>(
        format, width, height, width * depth, copy->flat<uint8_t>().data(),
        [copy](uint8*) { delete copy; });
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected float or uint8 tensor, received ",
                     DataTypeString(input_tensor.dtype())));
  }

  cc->Outputs().Tag(kImage).Add(output.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

}  // namespace mediapipe
