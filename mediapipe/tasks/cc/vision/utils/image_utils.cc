/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/tensor.h"
#include "stb_image.h"

namespace mediapipe {
namespace tasks {
namespace vision {

absl::StatusOr<Image> DecodeImageFromFile(const std::string& path) {
  int width;
  int height;
  int channels;
  auto* image_data = stbi_load(path.c_str(), &width, &height, &channels,
                               /*desired_channels=*/0);
  if (image_data == nullptr) {
    return absl::InternalError(absl::StrFormat("Image decoding failed (%s): %s",
                                               stbi_failure_reason(), path));
  }
  ImageFrameSharedPtr image_frame;
  switch (channels) {
    case 1:
      image_frame =
          std::make_shared<ImageFrame>(ImageFormat::GRAY8, width, height, width,
                                       image_data, stbi_image_free);
      break;
    case 3:
      image_frame =
          std::make_shared<ImageFrame>(ImageFormat::SRGB, width, height,
                                       3 * width, image_data, stbi_image_free);
      break;
    case 4:
      image_frame =
          std::make_shared<ImageFrame>(ImageFormat::SRGBA, width, height,
                                       4 * width, image_data, stbi_image_free);
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected image with 1 (grayscale), 3 (RGB) or 4 "
                          "(RGBA) channels, found %d channels.",
                          channels));
  }
  return Image(std::move(image_frame));
}

absl::StatusOr<Image> CreateImageFromBuffer(ImageFormat::Format format,
                                            const uint8_t* pixel_data,
                                            int width, int height) {
  int width_step = 0;
  switch (format) {
    case ImageFormat::GRAY8:
      width_step = width;
      break;
    case ImageFormat::SRGB:
      width_step = 3 * width;
      break;
    case ImageFormat::SRGBA:
      width_step = 4 * width;
      break;
    case ImageFormat::SBGRA:
      width_step = 4 * width;
      break;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected image of SRGB, SRGBA or SBGRA format, but found %d.",
          format));
  }
  ImageFrameSharedPtr image_frame = std::make_shared<ImageFrame>(
      format, width, height, width_step, const_cast<uint8_t*>(pixel_data),
      ImageFrame::PixelDataDeleter::kNone);
  return Image(std::move(image_frame));
}

absl::StatusOr<Shape> GetImageLikeTensorShape(const mediapipe::Tensor& tensor) {
  int width = 0;
  int height = 0;
  int channels = 1;
  const int dims_size = tensor.shape().dims.size();
  switch (dims_size) {
    case 2: {
      height = tensor.shape().dims[0];
      width = tensor.shape().dims[1];
      break;
    }
    case 3: {
      height = tensor.shape().dims[0];
      width = tensor.shape().dims[1];
      channels = tensor.shape().dims[2];
      break;
    }
    case 4: {
      height = tensor.shape().dims[1];
      width = tensor.shape().dims[2];
      channels = tensor.shape().dims[3];
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor should have 2, 3, or 4 dims, received: %d", dims_size));
  }
  return {{height, width, channels}};
}

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
