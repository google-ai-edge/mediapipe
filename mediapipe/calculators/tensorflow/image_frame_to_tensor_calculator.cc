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

#include <memory>

#include "mediapipe/calculators/tensorflow/image_frame_to_tensor_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace mediapipe {

namespace tf = tensorflow;

namespace {
// Convert the ImageFrame into Tensor with floating point value type.
// The value will be normalized based on mean and stddev.
std::unique_ptr<tf::Tensor> ImageFrameToNormalizedTensor(
    const ImageFrame& image_frame, float mean, float stddev) {
  const int cols = image_frame.Width();
  const int rows = image_frame.Height();
  const int channels = image_frame.NumberOfChannels();
  const uint8* pixel = image_frame.PixelData();
  const int width_padding = image_frame.WidthStep() - cols * channels;
  auto tensor = ::absl::make_unique<tf::Tensor>(
      tf::DT_FLOAT, tf::TensorShape({rows, cols, channels}));
  auto tensor_data = tensor->tensor<float, 3>();

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      for (int channel = 0; channel < channels; ++channel) {
        tensor_data(row, col, channel) = (pixel[channel] - mean) / stddev;
      }
      pixel += channels;
    }
    pixel += width_padding;
  }
  return tensor;
}

}  // namespace

// Converts ImageFrames to TensorFlow Tensors.
//
// The calculator expects one input (a packet containing an ImageFrame) and
// generates one output (a packet containing a tf::Tensor holding the same
// pixel data). The output tensor will be 3D with dimensions corresponding to
// height, width, and the number of channels (e.g. 3 for RGB or 1 for GRAY8).
//
// This calculator supports ImageFrame objects with any valid format (SRGB
// SRGBA, GRAY8, GRAY16, and VEC32F1). It will generate a Tensor using DT_UINT8
// for the first three types, DT_UINT16 for GRAY16, and DT_FLOAT for VEC32F1.
//
// The ImageFrame data can be packed or padded. The pixel data will be copied
// to the Tensor in row-major order.
//
// Example config:
//  node {
//    calculator: "ImageFrameToTensorCalculator"
//    input_stream: "scaled_frames"
//    output_stream: "video_tensors"
//  }
class ImageFrameToTensorCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  ImageFrameToTensorCalculatorOptions options_;
};
REGISTER_CALCULATOR(ImageFrameToTensorCalculator);

::mediapipe::Status ImageFrameToTensorCalculator::GetContract(
    CalculatorContract* cc) {
  // Start with only one input packet.
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one input stream is supported.";
  cc->Inputs().Index(0).Set<ImageFrame>(
      // ImageFrame frame.
  );
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1)
      << "Only one output stream is supported.";
  cc->Outputs().Index(0).Set<tf::Tensor>(
      // Output TensorFlow Tensor.
  );
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageFrameToTensorCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<ImageFrameToTensorCalculatorOptions>();
  // Inform the framework that we always output at the same timestamp
  // as we receive a packet at.
  cc->SetOffset(TimestampDiff(0));
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ImageFrameToTensorCalculator::Process(
    CalculatorContext* cc) {
  const Packet& input_item = cc->Inputs().Index(0).Value();
  RET_CHECK(!input_item.IsEmpty()) << "Input cannot be empty.";

  // Extract the ImageFrame and metadata from the input packet.
  const ImageFrame& video_frame = input_item.Get<ImageFrame>();
  const int bytes_per_pixel = video_frame.ByteDepth();

  std::unique_ptr<tf::Tensor> tensor;
  if (options_.has_data_type()) {
    RET_CHECK_EQ(bytes_per_pixel, 1) << "Unsupported image format ("
                                     << bytes_per_pixel << " bytes per pixel)";
    const tf::DataType data_type = options_.data_type();
    RET_CHECK_EQ(data_type, tf::DT_FLOAT)
        << "Unsupported data type " << data_type;
    RET_CHECK_GT(options_.stddev(), 0.0f);
    tensor = ImageFrameToNormalizedTensor(video_frame, options_.mean(),
                                          options_.stddev());
  } else {
    const int height = video_frame.Height();
    const int width = video_frame.Width();
    const int num_channels = video_frame.NumberOfChannels();
    const int num_components = width * height * num_channels;
    tf::TensorShape tensor_shape({height, width, num_channels});

    // Use uint8 uint16, or float as the TF type depending on bpp of ImageFrame.
    tf::DataType data_type;
    if (bytes_per_pixel == 1) {
      data_type = tf::DT_UINT8;
    } else if (bytes_per_pixel == 2) {
      data_type = tf::DT_UINT16;
    } else if (bytes_per_pixel == 4) {
      data_type = tf::DT_FLOAT;
    } else {
      return ::mediapipe::InvalidArgumentError(absl::StrCat(
          "Unsupported image format (", bytes_per_pixel, " bytes per pixel)"));
    }

    // This failure should never trigger, but it protects the code against
    // internal TF changes.
    RET_CHECK(tf::DataTypeCanUseMemcpy(data_type))
        << "Tensor data type does not support memcpy (type=" << data_type
        << ")";

    // Create the output tensor.
    tensor = ::absl::make_unique<tf::Tensor>(data_type, tensor_shape);

    // Copy pixel data from the ImageFrame to the tensor.
    if (data_type == tf::DT_UINT8) {
      uint8* dst = tensor->flat<uint8>().data();
      video_frame.CopyToBuffer(dst, num_components);
    } else if (data_type == tf::DT_UINT16) {
      uint16* dst = tensor->flat<uint16>().data();
      video_frame.CopyToBuffer(dst, num_components);
    } else {
      float* dst = tensor->flat<float>().data();
      video_frame.CopyToBuffer(dst, num_components);
    }
  }

  cc->Outputs().Index(0).Add(tensor.release(), cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
