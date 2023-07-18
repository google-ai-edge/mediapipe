// Copyright (c) 2023 Intel Corporation
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
//

#include <string>
#include <vector>

#include "mediapipe/calculators/openvino/openvino_converter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"

#include <openvino/openvino.hpp>

namespace {
constexpr char kImageFrameTag[] = "IMAGE";
constexpr char kGpuBufferTag[] = "IMAGE_GPU";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kRemoteTensorsTag[] = "TENSORS_REMOTE";
}  // namespace

namespace mediapipe {

// Calculator for normalizing and converting an ImageFrame or Matrix
// into a ov::Tensor or a GpuBuffer to a ov::RemoteTensor
//
// This calculator is designed to be used with the OpenVINOInferenceCalculator,
// as a pre-processing step for calculator inputs.
//
// IMAGE and IMAGE_GPU inputs are normalized to [-1,1] (default) or [0,1],
// specified by options (unless outputting a quantized tensor).
//
// Input:
//  One of the following tags:
//  IMAGE - ImageFrame (assumed to be 8-bit or 32-bit data).
//  IMAGE_GPU - GpuBuffer (assumed to be RGBA or RGB GL texture).
//
// Output:
//  One of the following tags:
//  TENSORS - Vector of ov::Tensors
//  TENSORS_REMOTE - Vector of ov::RemoteTensors
//
// Example use:
// node {
//   calculator: "OpenVINOConverterCalculator"
//   input_stream: "IMAGE:input_image"
//   output_stream: "TENSORS:image_tensor"
//   options: {
//     [mediapipe.OpenVINOConverterCalculatorOptions.ext] {
//       zero_center: true
//     }
//   }
// }
//
// IMPORTANT Notes:
//  No conversion between CPU/GPU is done.
//  Inputs/outputs must match type: CPU->CPU or GPU->GPU.
//  This calculator uses FixedSizeInputStreamHandler by default.
//
// Note: Input defines output, so only these type sets are supported:
// IMAGE -> TENSORS | IMAGE_GPU -> TENSORS_GPU 
//
class OpenVINOConverterCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status LoadOptions(CalculatorContext* cc);
  template <class T>
  absl::Status NormalizeImage(const ImageFrame& image_frame,
                              bool flip_vertically, float* tensor_ptr);

  bool initialized_ = false;
  absl::optional<std::pair<float, float>> output_range_;
  bool flip_vertically_ = false;
  int max_num_channels_ = 3;
};
REGISTER_CALCULATOR(OpenVINOConverterCalculator);

namespace {
template <class CC>
bool ShouldUseGpu(CC* cc) {
#if MEDIAPIPE_OPENVINO_GPU_SUPPORTED
  return cc->Inputs().HasTag(kGpuBufferTag) ||
         cc->Outputs().HasTag(kTensorsGpuTag);
#else
  return false;
#endif  // MEDIAPIPE_OPENVINO_GPU_SUPPORTED
}
}  // namespace

absl::Status OpenVINOConverterCalculator::GetContract(CalculatorContract* cc) {
  // Confirm only one of the input streams is present.
  RET_CHECK(cc->Inputs().HasTag(kImageFrameTag) ^
            cc->Inputs().HasTag(kGpuBufferTag));

  // Confirm only one of the output streams is present.
  RET_CHECK(cc->Outputs().HasTag(kTensorsTag) ^
            cc->Outputs().HasTag(kRemoteTensorsTag));

  if (cc->Inputs().HasTag(kImageFrameTag)) {
    cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
  }
//#if !MEDIAPIPE_DISABLE_GPU
//  if (cc->Inputs().HasTag(kGpuBufferTag)) {
//    cc->Inputs().Tag(kGpuBufferTag).Set<mediapipe::GpuBuffer>();
//  }
//#endif  // !MEDIAPIPE_DISABLE_GPU

  if (cc->Outputs().HasTag(kTensorsTag)) {
    cc->Outputs().Tag(kTensorsTag).Set<std::vector<ov::Tensor>>();
  }
  if (cc->Outputs().HasTag(kRemoteTensorsTag)) {
    cc->Outputs().Tag(kRemoteTensorsTag).Set<std::vector<ov::RemoteTensor>>();
  }

  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return absl::OkStatus();
}

absl::Status OpenVINOConverterCalculator::Open(CalculatorContext* cc) {
  // Inform the framework that we always output at the same timestamp
  // as we receive a packet at.
  cc->SetOffset(TimestampDiff(0));

  MP_RETURN_IF_ERROR(LoadOptions(cc));

  return absl::OkStatus();
}

absl::Status OpenVINOConverterCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

absl::Status OpenVINOConverterCalculator::Process(CalculatorContext* cc) {
  RET_CHECK(cc->Inputs().HasTag(kImageFrameTag)) << "Only supporting ImageFrame inputs at the moment";

  if (cc->Inputs().HasTag(kImageFrameTag)) {
    if (cc->Inputs().Tag(kImageFrameTag).IsEmpty()) {
      return absl::OkStatus();
    }
    const auto &input_item = cc->Inputs().Tag(kImageFrameTag);
    RET_CHECK(!input_item.IsEmpty()) << "Input cannot be empty.";

    // Extract the ImageFrame and metadata from the input packet.
    const ImageFrame &image_frame = input_item.Get<ImageFrame>();
    const int bytes_per_pixel = image_frame.ByteDepth();

    const size_t height = image_frame.Height();
    const size_t width = image_frame.Width();
    const size_t num_channels = image_frame.NumberOfChannels();
    const size_t num_components = width * height * num_channels;
    ov::Shape tensor_shape({1, height, width, num_channels});

    auto output_tensors = ::absl::make_unique<std::vector<ov::Tensor>>();

    if (output_range_.has_value()) {
      // Normalize
      ov::Tensor tensor(ov::element::f32, tensor_shape);
      float* tensor_buffer = tensor.data<float>();
      RET_CHECK(tensor_buffer);

      if (image_frame.ByteDepth() == 1) {
        MP_RETURN_IF_ERROR(NormalizeImage<uint8>(image_frame, flip_vertically_,
                                                 tensor_buffer));
      } else if (image_frame.ByteDepth() == 4) {
        MP_RETURN_IF_ERROR(NormalizeImage<float>(image_frame, flip_vertically_,
                                                 tensor_buffer));
      } else {
        return absl::InternalError(
                "Only byte-based (8 bit) and float (32 bit) images supported.");
      }
      output_tensors->emplace_back(tensor);
    } else {
      // Copy ImageFrame data directly into OpenVINO Tensor, element type is preserved.
      ov::element::Type data_type;
      if (bytes_per_pixel == 1) {
        data_type = ov::element::u8;
      } else if (bytes_per_pixel == 2) {
        data_type = ov::element::u16;
      } else if (bytes_per_pixel == 4) {
        data_type = ov::element::f32;
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
                "Unsupported image format (", bytes_per_pixel, " bytes per pixel)"));
      }

      // Create the output tensor
      // TODO: is zero-copy possible?
      ov::Tensor tensor(data_type, tensor_shape);
      // Copy pixel data from the ImageFrame to the tensor.
      // TODO: what if there are strides in the tensor?
      if (data_type == ov::element::u8) {
        uint8 *dst = (uint8 *) tensor.data();
        image_frame.CopyToBuffer(dst, num_components);
      } else if (data_type == ov::element::u16) {
        uint16 *dst = (uint16_t *) tensor.data();
        image_frame.CopyToBuffer(dst, num_components);
      } else {
        float *dst = (float *) tensor.data();
        image_frame.CopyToBuffer(dst, num_components);
      }
      output_tensors->emplace_back(tensor);
    }

    cc->Outputs()
            .Tag(kTensorsTag)
            .Add(output_tensors.release(), cc->InputTimestamp());
  }
  // TODO: else if (kGPUBufferTag)

  return absl::OkStatus();
}

absl::Status OpenVINOConverterCalculator::LoadOptions(CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  const auto& options =
      cc->Options<::mediapipe::OpenVINOConverterCalculatorOptions>();

  if (options.enable_normalization()) {
    // if zero_center, set output float range to match [-1, 1] as specified in
    // calculator proto.
    if (options.zero_center()) {
      output_range_.emplace(std::pair<float, float>(-1.0, 1.0));
    }

    // Custom output_tensor_float_range values.
    // If the float range is specified in pb text, use the specified values
    // instead.
    if (options.has_output_tensor_float_range()) {
      output_range_.emplace(options.output_tensor_float_range().min(),
                            options.output_tensor_float_range().max());
      CHECK_GT(output_range_->second, output_range_->first);
    }

    // Custom div and sub values.
    if (options.use_custom_normalization()) {
      output_range_.emplace(std::pair<float, float>(
              -options.custom_sub(),
              -options.custom_sub() + 255.0 / options.custom_div()));
    }
  }

  // Get y-flip mode.
  flip_vertically_ = options.flip_vertically();

  // Get desired way to handle input channels.
  max_num_channels_ = options.max_num_channels();
  CHECK_GE(max_num_channels_, 1);
  CHECK_LE(max_num_channels_, 4);
  CHECK_NE(max_num_channels_, 2);

  return absl::OkStatus();
}

template <class T>
absl::Status OpenVINOConverterCalculator::NormalizeImage(
    const ImageFrame& image_frame, bool flip_vertically, float* tensor_ptr) {
  const int height = image_frame.Height();
  const int width = image_frame.Width();
  const int channels = image_frame.NumberOfChannels();
  const int channels_preserved = std::min(channels, max_num_channels_);
  const int channels_ignored = channels - channels_preserved;

  if (output_range_.has_value()) {
    // If the output float range is set and we are not using custom
    // normalization, normalize the pixel values from [0, 255] to the specified
    // output range.
    RET_CHECK_NE(output_range_->first, output_range_->second);
    const float scale = (output_range_->second - output_range_->first) / 255.0f;
    const float bias = output_range_->first;

    for (int i = 0; i < height; ++i) {
      const T* image_ptr = reinterpret_cast<const T*>(
          image_frame.PixelData() +
          (flip_vertically ? height - 1 - i : i) * image_frame.WidthStep());
      for (int j = 0; j < width; ++j) {
        for (int c = 0; c < channels_preserved; ++c) {
          *tensor_ptr++ = *image_ptr++ * scale + bias;
        }
        image_ptr += channels_ignored;
      }
    }
  } else {
    // [0,1], scale only (bias == 0)
    // Verified that there are no precision issues with 1.0f / 255.0f expression
    const float scale = 1.0f / 255.0f;
    for (int i = 0; i < height; ++i) {
      const T* image_ptr = reinterpret_cast<const T*>(
          image_frame.PixelData() +
          (flip_vertically ? height - 1 - i : i) * image_frame.WidthStep());
      for (int j = 0; j < width; ++j) {
        for (int c = 0; c < channels_preserved; ++c) {
          *tensor_ptr++ = *image_ptr++ * scale;
        }
        image_ptr += channels_ignored;
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace mediapipe
