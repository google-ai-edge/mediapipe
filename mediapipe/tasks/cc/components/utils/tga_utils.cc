/* Copyright 2026 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/components/utils/tga_utils.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/formats/image.h"

namespace mediapipe::tasks::components::utils {
namespace {

constexpr size_t kTgaHeaderBytes = 18;

}  // namespace

absl::StatusOr<std::string> EncodeToTga(const mediapipe::Image& image) {
  auto image_frame = image.GetImageFrameSharedPtr();
  if (image_frame == nullptr) {
    return absl::InvalidArgumentError("Image cannot be converted to CPU.");
  }
  const int width = image_frame->Width();
  const int height = image_frame->Height();
  const int channels = image_frame->NumberOfChannels();
  if (channels != 1 && channels != 3 && channels != 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported number of image channels: ", channels));
  }

  uint8_t header[kTgaHeaderBytes] = {0};
  header[2] = (channels == 1) ? 3 : 2;
  header[12] = static_cast<uint8_t>(width & 0xFF);
  header[13] = static_cast<uint8_t>((width >> 8) & 0xFF);
  header[14] = static_cast<uint8_t>(height & 0xFF);
  header[15] = static_cast<uint8_t>((height >> 8) & 0xFF);
  header[16] = static_cast<uint8_t>(channels * 8);
  header[17] = (channels == 4) ? 0x28 : 0x20;

  const size_t pixel_size =
      static_cast<size_t>(width) * static_cast<size_t>(height) * channels;
  std::string tga_bytes(kTgaHeaderBytes + pixel_size, '\0');
  std::memcpy(&tga_bytes[0], header, kTgaHeaderBytes);
  uint8_t* dst_pixels = reinterpret_cast<uint8_t*>(&tga_bytes[kTgaHeaderBytes]);
  const uint8_t* pixel_data = image_frame->PixelData();
  const int width_step = image_frame->WidthStep();
  const int row_size = width * channels;

  // Standard TGA specifications expect BGR or BGRA ordering (Blue, Green, Red).
  for (int y = 0; y < height; ++y) {
    const uint8_t* src_row = pixel_data + y * width_step;
    uint8_t* dst_row = dst_pixels + y * row_size;
    if (channels == 3 || channels == 4) {
      for (int x = 0; x < width; ++x) {
        dst_row[x * channels + 0] = src_row[x * channels + 2];
        dst_row[x * channels + 1] = src_row[x * channels + 1];
        dst_row[x * channels + 2] = src_row[x * channels + 0];
        if (channels == 4) {
          dst_row[x * 4 + 3] = src_row[x * 4 + 3];
        }
      }
    } else {
      std::memcpy(dst_row, src_row, row_size);
    }
  }
  return tga_bytes;
}

}  // namespace mediapipe::tasks::components::utils
