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

#include "mediapipe/tasks/c/vision/core/image.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "stb_image.h"

namespace {

using ::mediapipe::Image;
using ::mediapipe::ImageFormat;
using ::mediapipe::ImageFrame;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::vision::core::GetCachedContiguousDataAttr;
using ::mediapipe::tasks::vision::core::GetValue;

using ImageFrameSharedPtr = std::shared_ptr<ImageFrame>;

// Converts C enum to C++ enum.
ImageFormat::Format ToCppImageFormat(MpImageFormat format) {
  switch (format) {
    case kMpImageFormatGray8:
      return ImageFormat::GRAY8;
    case kMpImageFormatGray16:
      return ImageFormat::GRAY16;
    case kMpImageFormatSrgb:
      return ImageFormat::SRGB;
    case kMpImageFormatSrgb48:
      return ImageFormat::SRGB48;
    case kMpImageFormatSrgba:
      return ImageFormat::SRGBA;
    case kMpImageFormatSrgba64:
      return ImageFormat::SRGBA64;
    case kMpImageFormatVec32F1:
      return ImageFormat::VEC32F1;
    case kMpImageFormatVec32F2:
      return ImageFormat::VEC32F2;
    case kMpImageFormatVec32F4:
      return ImageFormat::VEC32F4;
    default:
      return ImageFormat::UNKNOWN;
  }
}

// Converts C enum to string.
std::string ToString(MpImageFormat format) {
  switch (format) {
    case kMpImageFormatUnknown:
      return "UNKNOWN";
    case kMpImageFormatGray8:
      return "GRAY8";
    case kMpImageFormatGray16:
      return "GRAY16";
    case kMpImageFormatSrgb:
      return "SRGB";
    case kMpImageFormatSrgb48:
      return "SRGB48";
    case kMpImageFormatSrgba:
      return "SRGBA";
    case kMpImageFormatSrgba64:
      return "SRGBA64";
    case kMpImageFormatVec32F1:
      return "VEC32F1";
    case kMpImageFormatVec32F2:
      return "VEC32F2";
    case kMpImageFormatVec32F4:
      return "VEC32F4";
    default:
      return "UNKNOWN";
  }
}

// Converts C++ enum to C enum.
MpImageFormat ToCImageFormat(ImageFormat::Format format) {
  switch (format) {
    case ImageFormat::GRAY8:
      return kMpImageFormatGray8;
    case ImageFormat::GRAY16:
      return kMpImageFormatGray16;
    case ImageFormat::SRGB:
      return kMpImageFormatSrgb;
    case ImageFormat::SRGB48:
      return kMpImageFormatSrgb48;
    case ImageFormat::SRGBA:
      return kMpImageFormatSrgba;
    case ImageFormat::SRGBA64:
      return kMpImageFormatSrgba64;
    case ImageFormat::VEC32F1:
      return kMpImageFormatVec32F1;
    case ImageFormat::VEC32F2:
      return kMpImageFormatVec32F2;
    case ImageFormat::VEC32F4:
      return kMpImageFormatVec32F4;
    default:
      return kMpImageFormatUnknown;
  }
}

absl::StatusOr<ImageFormat::Format> GetImageFormatFromChannels(int channels) {
  switch (channels) {
    case 1:
      return ImageFormat::GRAY8;
    case 3:
      return ImageFormat::SRGB;
    case 4:
      return ImageFormat::SRGBA;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected image with 1 (grayscale), 3 (RGB) or 4 (RGBA) channels, "
          "found %d channels.",
          channels));
  }
}

ImageFrameSharedPtr GetImageFrameSharedPtr(MpImagePtr image) {
  return image->image.GetImageFrameSharedPtr();
}

absl::Status ValidateDimensions(const MpImagePtr image, int pos_size) {
  if (pos_size != 3 &&
      !(pos_size == 2 &&
        GetImageFrameSharedPtr(image)->NumberOfChannels() == 1)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid index dimension: %d (expected 2 or 3)", pos_size));
  }
  return absl::OkStatus();
}

// Helper to create and initialize an MpImage from pixel data.
template <typename T>
absl::Status CreateMpImageInternal(MpImageFormat format, int width, int height,
                                   const T* pixel_data, int pixel_data_size,
                                   MpImagePtr* out) {
  auto image_frame =
      std::make_shared<ImageFrame>(ToCppImageFormat(format), width, height,
                                   ImageFrame::kDefaultAlignmentBoundary);

  // Validate pixel_data_size.
  const int expected_min_size = image_frame->Height() * image_frame->Width() *
                                image_frame->NumberOfChannels() *
                                image_frame->ByteDepth();
  if (pixel_data_size < expected_min_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Pixel data size is too small: %d (expected %d)",
                        pixel_data_size, expected_min_size));
  }

  const int row_size_bytes = image_frame->Width() *
                             image_frame->NumberOfChannels() *
                             image_frame->ByteDepth();
  if (image_frame->WidthStep() == row_size_bytes) {
    std::memcpy(image_frame->MutablePixelData(), pixel_data,
                image_frame->Height() * image_frame->WidthStep());
  } else {
    for (int i = 0; i < image_frame->Height(); ++i) {
      std::memcpy(
          image_frame->MutablePixelData() + i * image_frame->WidthStep(),
          reinterpret_cast<const uint8_t*>(pixel_data) + i * row_size_bytes,
          row_size_bytes);
    }
  }

  auto image = std::make_unique<MpImageInternal>();
  image->image = Image(std::move(image_frame));
  *out = image.release();
  return absl::OkStatus();
}

}  // namespace

extern "C" {

MP_EXPORT MpStatus MpImageCreateFromUint8Data(MpImageFormat format, int width,
                                              int height,
                                              const uint8_t* pixel_data,
                                              int pixel_data_size,
                                              MpImagePtr* out) {
  if (!pixel_data) {
    ABSL_LOG(ERROR) << "Pixel data is null";
    return kMpInvalidArgument;
  }

  if (format != kMpImageFormatGray8 && format != kMpImageFormatSrgb &&
      format != kMpImageFormatSrgba) {
    ABSL_LOG(ERROR) << "Unsupported image format: " << ToString(format)
                    << " (expected GRAY8, SRGB, or SRGBA for uint8_t data)";
    return kMpInvalidArgument;
  }

  auto status = CreateMpImageInternal(format, width, height, pixel_data,
                                      pixel_data_size, out);
  return ToMpStatus(status);
}

MP_EXPORT MpStatus MpImageCreateFromUint16Data(MpImageFormat format, int width,
                                               int height,
                                               const uint16_t* pixel_data,
                                               int pixel_data_size,
                                               MpImagePtr* out) {
  if (!pixel_data) {
    ABSL_LOG(ERROR) << "Pixel data is null";
    return kMpInvalidArgument;
  }

  if (format != kMpImageFormatGray16 && format != kMpImageFormatSrgb48 &&
      format != kMpImageFormatSrgba64) {
    ABSL_LOG(ERROR)
        << "Unsupported image format: " << ToString(format)
        << " (expected GRAY16, SRGB48, or SRGBA64 for uint16_t data)";
    return kMpInvalidArgument;
  }

  auto status = CreateMpImageInternal(format, width, height, pixel_data,
                                      pixel_data_size * sizeof(uint16_t), out);
  return ToMpStatus(status);
}

MP_EXPORT MpStatus MpImageCreateFromFloatData(MpImageFormat format, int width,
                                              int height,
                                              const float* pixel_data,
                                              int pixel_data_size,
                                              MpImagePtr* out) {
  if (!pixel_data) {
    ABSL_LOG(ERROR) << "Pixel data is null";
    return kMpInvalidArgument;
  }

  if (format != kMpImageFormatVec32F1 && format != kMpImageFormatVec32F2 &&
      format != kMpImageFormatVec32F4) {
    ABSL_LOG(ERROR)
        << "Unsupported image format: " << ToString(format)
        << " (expected VEC32F1, VEC32F2, or VEC32F4 for float data)";
    return kMpInvalidArgument;
  }

  auto status = CreateMpImageInternal(format, width, height, pixel_data,
                                      pixel_data_size * sizeof(float), out);
  return ToMpStatus(status);
}

MP_EXPORT MpStatus MpImageCreateFromFile(const char* file_name,
                                         MpImagePtr* out) {
  unsigned char* image_data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 0;

#if TARGET_OS_OSX && !MEDIAPIPE_DISABLE_GPU
  // On MacOS, stbi_load does not support 3-channel images, so we read the
  // number of channels first and request RGBA if needed.
  if (stbi_info(file_name, &width, &height, &channels)) {
    if (channels == 3) {
      channels = 4;
    }
    int unused;
    image_data = stbi_load(file_name, &width, &height, &unused, channels);
  }
#else
  image_data = stbi_load(file_name, &width, &height, &channels,
                         /*desired_channels=*/0);
#endif  // TARGET_OS_OSX && !MEDIAPIPE_DISABLE_GPU
  if (image_data == nullptr) {
    ABSL_LOG(ERROR) << "Failed to load image from file: " << file_name;
    return kMpInternal;
  }

  auto format = GetImageFormatFromChannels(channels);
  if (!format.ok()) {
    ABSL_LOG(ERROR) << "Unsupported image format: " << format.status();
    stbi_image_free(image_data);
    return kMpInvalidArgument;
  }

  auto image = std::make_shared<ImageFrame>(
      *format, width, height, channels * width, image_data, stbi_image_free);

  auto mp_image = std::make_unique<MpImageInternal>();
  mp_image->image = Image(std::move(image));
  *out = mp_image.release();
  return kMpOk;
}

MP_EXPORT MpStatus MpImageCreateFromImageFrame(MpImagePtr image,
                                               MpImagePtr* out) {
  auto mp_image = std::make_unique<MpImageInternal>();
  mp_image->image = Image(image->image.GetImageFrameSharedPtr());
  *out = mp_image.release();
  return kMpOk;
}

MP_EXPORT bool MpImageUsesGpu(const MpImagePtr image) {
  return image->image.UsesGpu();
}

MP_EXPORT bool MpImageIsContiguous(const MpImagePtr image) {
  return GetImageFrameSharedPtr(image)->IsContiguous();
}

MP_EXPORT bool MpImageIsEmpty(const MpImagePtr image) {
  return GetImageFrameSharedPtr(image)->IsEmpty();
}

MP_EXPORT bool MpImageIsAligned(const MpImagePtr image,
                                uint32_t alignment_boundary) {
  return GetImageFrameSharedPtr(image)->IsAligned(alignment_boundary);
}

MP_EXPORT int MpImageGetWidth(const MpImagePtr image) {
  return GetImageFrameSharedPtr(image)->Width();
}

MP_EXPORT int MpImageGetHeight(const MpImagePtr image) {
  return GetImageFrameSharedPtr(image)->Height();
}

MP_EXPORT int MpImageGetChannels(const MpImagePtr image) {
  return GetImageFrameSharedPtr(image)->NumberOfChannels();
}

MP_EXPORT int MpImageGetByteDepth(const MpImagePtr image) {
  return GetImageFrameSharedPtr(image)->ByteDepth();
}

MP_EXPORT int MpImageGetWidthStep(const MpImagePtr image) {
  return GetImageFrameSharedPtr(image)->WidthStep();
}

MP_EXPORT MpImageFormat MpImageGetFormat(const MpImagePtr image) {
  return ToCImageFormat(GetImageFrameSharedPtr(image)->Format());
}

MP_EXPORT MpStatus MpImageDataUint8(const MpImagePtr image,
                                    const uint8_t** out) {
  auto data = GetCachedContiguousDataAttr<uint8_t>(image);
  if (data.ok()) {
    *out = *data;
  }
  return ToMpStatus(data.status());
}

MP_EXPORT MpStatus MpImageDataUint16(const MpImagePtr image,
                                     const uint16_t** out) {
  auto data = GetCachedContiguousDataAttr<uint16_t>(image);
  if (data.ok()) {
    *out = *data;
  }
  return ToMpStatus(data.status());
}

MP_EXPORT MpStatus MpImageDataFloat32(const MpImagePtr image,
                                      const float** out) {
  auto data = GetCachedContiguousDataAttr<float>(image);
  if (data.ok()) {
    *out = *data;
  }
  return ToMpStatus(data.status());
}

MP_EXPORT MpStatus MpImageGetValueUint8(const MpImagePtr image, int* pos,
                                        int pos_size, uint8_t* out) {
  auto status = ValidateDimensions(image, pos_size);
  if (!status.ok()) {
    return ToMpStatus(status);
  }

  size_t byte_depth = GetImageFrameSharedPtr(image)->ByteDepth();
  if (byte_depth != 1) {
    ABSL_LOG(ERROR) << "Unexpected image byte depth: " << byte_depth
                    << " (expected 1 for uint8_t data)";
    return kMpInvalidArgument;
  }

  std::vector<int> pos_vec(pos, pos + pos_size);
  auto value = GetValue<uint8_t>(image, pos_vec);
  if (value.ok()) {
    *out = *value;
  }
  return ToMpStatus(value.status());
}

MP_EXPORT MpStatus MpImageGetValueUint16(const MpImagePtr image, int* pos,
                                         int pos_size, uint16_t* out) {
  auto status = ValidateDimensions(image, pos_size);
  if (!status.ok()) {
    return ToMpStatus(status);
  }

  size_t byte_depth = GetImageFrameSharedPtr(image)->ByteDepth();
  if (byte_depth != 2) {
    ABSL_LOG(ERROR) << "Unexpected image byte depth: " << byte_depth
                    << " (expected 2 for uint16_t data)";
    return kMpInvalidArgument;
  }

  std::vector<int> pos_vec(pos, pos + pos_size);
  auto value = GetValue<uint16_t>(image, pos_vec);
  if (value.ok()) {
    *out = *value;
  }
  return ToMpStatus(value.status());
}

MP_EXPORT MpStatus MpImageGetValueFloat32(const MpImagePtr image, int* pos,
                                          int pos_size, float* out) {
  auto status = ValidateDimensions(image, pos_size);
  if (!status.ok()) {
    return ToMpStatus(status);
  }

  size_t byte_depth = GetImageFrameSharedPtr(image)->ByteDepth();
  if (byte_depth != 4) {
    ABSL_LOG(ERROR) << "Unexpected image byte depth: " << byte_depth
                    << " (expected 4 for float data)";
    return kMpInvalidArgument;
  }

  std::vector<int> pos_vec(pos, pos + pos_size);
  auto value = GetValue<float>(image, pos_vec);
  if (value.ok()) {
    *out = *value;
  }
  return ToMpStatus(value.status());
}

MP_EXPORT void MpImageFree(MpImagePtr image) { delete image; }

}  // extern "C"
