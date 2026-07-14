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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "exif.h"
#include "mediapipe/framework/deps/file_helpers.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/c/vision/core/image_frame_util.h"
#include "stb_image.h"

namespace {

// EXIF orientation values (1-8)
constexpr uint16_t kExifOrientationNormal = 1;
constexpr uint16_t kExifOrientationMirrorHorizontal = 2;
constexpr uint16_t kExifOrientationRotate180 = 3;
constexpr uint16_t kExifOrientationMirrorVertical = 4;
constexpr uint16_t kExifOrientationMirrorHorizontalRotate270CW = 5;
constexpr uint16_t kExifOrientationRotate90CW = 6;
constexpr uint16_t kExifOrientationMirrorHorizontalRotate90CW = 7;
constexpr uint16_t kExifOrientationRotate270CW = 8;

using ::mediapipe::Image;
using ::mediapipe::ImageFormat;
using ::mediapipe::ImageFrame;
using ::mediapipe::tasks::c::core::HandleStatus;
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

// Applies EXIF orientation to the loaded image bytes directly writing to a
// Creates an ImageFrame from loaded pixel data, applying EXIF orientation.
// If orientation <= 1, wraps the loaded stbi pixel data zero-copy and uses
// stbi_image_free deleter. If orientation > 1, creates a new aligned
// ImageFrame, rotates the pixels from the loaded data, and frees the original
// loaded data.
std::shared_ptr<ImageFrame> CreateImageFrameWithOrientation(
    unsigned char* image_data, int width, int height, int channels,
    ImageFormat::Format format, uint16_t orientation) {
  if (orientation <= kExifOrientationNormal ||
      orientation > kExifOrientationRotate270CW) {
    return std::make_shared<ImageFrame>(format, width, height, channels * width,
                                        image_data, stbi_image_free);
  }

  int dest_width = width;
  int dest_height = height;
  if (orientation >= kExifOrientationMirrorHorizontalRotate270CW &&
      orientation <= kExifOrientationRotate270CW) {
    dest_width = height;
    dest_height = width;
  }

  auto image = std::make_shared<ImageFrame>(
      format, dest_width, dest_height, ImageFrame::kDefaultAlignmentBoundary);

  unsigned char* dest_data = image->MutablePixelData();
  int dest_width_step = image->WidthStep();

  // Helper lambda to abstract pixel copying with width step / row stride
  auto copy_pixel = [&](int dest_x, int dest_y, int src_x, int src_y) {
    std::memcpy(&dest_data[dest_y * dest_width_step + dest_x * channels],
                &image_data[(src_y * width + src_x) * channels], channels);
  };

  switch (orientation) {
    case kExifOrientationMirrorHorizontal:
      for (int y = 0; y < dest_height; ++y) {
        for (int x = 0; x < dest_width; ++x) {
          copy_pixel(x, y, width - 1 - x, y);
        }
      }
      break;
    case kExifOrientationRotate180:
      for (int y = 0; y < dest_height; ++y) {
        for (int x = 0; x < dest_width; ++x) {
          copy_pixel(x, y, width - 1 - x, height - 1 - y);
        }
      }
      break;
    case kExifOrientationMirrorVertical:
      for (int y = 0; y < dest_height; ++y) {
        std::memcpy(&dest_data[y * dest_width_step],
                    &image_data[(height - 1 - y) * width * channels],
                    dest_width * channels);
      }
      break;
    case kExifOrientationMirrorHorizontalRotate270CW:
      for (int y = 0; y < dest_height; ++y) {
        for (int x = 0; x < dest_width; ++x) {
          copy_pixel(x, y, y, x);
        }
      }
      break;
    case kExifOrientationRotate90CW:
      for (int y = 0; y < dest_height; ++y) {
        for (int x = 0; x < dest_width; ++x) {
          copy_pixel(x, y, y, height - 1 - x);
        }
      }
      break;
    case kExifOrientationMirrorHorizontalRotate90CW:
      for (int y = 0; y < dest_height; ++y) {
        for (int x = 0; x < dest_width; ++x) {
          copy_pixel(x, y, width - 1 - y, height - 1 - x);
        }
      }
      break;
    case kExifOrientationRotate270CW:
      for (int y = 0; y < dest_height; ++y) {
        for (int x = 0; x < dest_width; ++x) {
          copy_pixel(x, y, width - 1 - y, x);
        }
      }
      break;
    default:
      break;
  }

  stbi_image_free(image_data);
  return image;
}

}  // namespace

extern "C" {

MP_EXPORT MpStatus MpImageCreateFromUint8Data(
    MpImageFormat format, int width, int height, const uint8_t* pixel_data,
    int pixel_data_size, MpImagePtr* out, char** error_msg) {
  if (!pixel_data) {
    return HandleStatus(absl::InvalidArgumentError("Pixel data is null"),
                        error_msg);
  }

  if (format != kMpImageFormatGray8 && format != kMpImageFormatSrgb &&
      format != kMpImageFormatSrgba) {
    return HandleStatus(
        absl::InvalidArgumentError(absl::StrFormat(
            "Unsupported image format: %s (expected GRAY8, SRGB, or "
            "SRGBA for uint8_t data)",
            ToString(format))),
        error_msg);
  }

  auto status = CreateMpImageInternal(format, width, height, pixel_data,
                                      pixel_data_size, out);
  return HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpImageCreateFromUint16Data(
    MpImageFormat format, int width, int height, const uint16_t* pixel_data,
    int pixel_data_size, MpImagePtr* out, char** error_msg) {
  if (!pixel_data) {
    return HandleStatus(absl::InvalidArgumentError("Pixel data is null"),
                        error_msg);
  }

  if (format != kMpImageFormatGray16 && format != kMpImageFormatSrgb48 &&
      format != kMpImageFormatSrgba64) {
    return HandleStatus(absl::InvalidArgumentError(absl::StrFormat(
                            "Unsupported image format: %s (expected GRAY16, "
                            "SRGB48, or SRGBA64 for uint16_t data)",
                            ToString(format))),
                        error_msg);
  }

  auto status = CreateMpImageInternal(format, width, height, pixel_data,
                                      pixel_data_size * sizeof(uint16_t), out);
  return HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpImageCreateFromFloatData(
    MpImageFormat format, int width, int height, const float* pixel_data,
    int pixel_data_size, MpImagePtr* out, char** error_msg) {
  if (!pixel_data) {
    return HandleStatus(absl::InvalidArgumentError("Pixel data is null"),
                        error_msg);
  }

  if (format != kMpImageFormatVec32F1 && format != kMpImageFormatVec32F2 &&
      format != kMpImageFormatVec32F4) {
    return HandleStatus(absl::InvalidArgumentError(absl::StrFormat(
                            "Unsupported image format: %s (expected VEC32F1, "
                            "VEC32F2, or VEC32F4 for float data)",
                            ToString(format))),
                        error_msg);
  }

  auto status = CreateMpImageInternal(format, width, height, pixel_data,
                                      pixel_data_size * sizeof(float), out);
  return HandleStatus(status, error_msg);
}

MP_EXPORT MpStatus MpImageCreateFromFile(const char* file_name, MpImagePtr* out,
                                         char** error_msg) {
  if (!file_name) {
    return HandleStatus(absl::InvalidArgumentError("File name is null"),
                        error_msg);
  }
  std::string file_contents;
  absl::Status file_status = ::mediapipe::file::GetContents(
      file_name, &file_contents, /*read_as_binary=*/true);
  if (!file_status.ok()) {
    return HandleStatus(file_status, error_msg);
  }

  uint16_t orientation = 1;
  easyexif::EXIFInfo exif_info;
  if (exif_info.parseFrom(
          reinterpret_cast<const unsigned char*>(file_contents.data()),
          file_contents.size()) == 0) {
    orientation = exif_info.Orientation;
  }

  unsigned char* image_data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 0;

#if TARGET_OS_OSX && !MEDIAPIPE_DISABLE_GPU
  // On MacOS, stbi_load does not support 3-channel images, so we read the
  // number of channels first and request RGBA if needed.
  if (stbi_info_from_memory(
          reinterpret_cast<const unsigned char*>(file_contents.data()),
          file_contents.size(), &width, &height, &channels)) {
    if (channels == 3) {
      channels = 4;
    }
    int unused;
    image_data = stbi_load_from_memory(
        reinterpret_cast<const unsigned char*>(file_contents.data()),
        file_contents.size(), &width, &height, &unused, channels);
  }
#else
  image_data = stbi_load_from_memory(
      reinterpret_cast<const unsigned char*>(file_contents.data()),
      file_contents.size(), &width, &height, &channels,
      /*desired_channels=*/0);
#endif  // TARGET_OS_OSX && !MEDIAPIPE_DISABLE_GPU
  if (image_data == nullptr) {
    return HandleStatus(absl::InternalError("Failed to load image from file"),
                        error_msg);
  }

  auto format = GetImageFormatFromChannels(channels);
  if (!format.ok()) {
    stbi_image_free(image_data);
    return HandleStatus(format.status(), error_msg);
  }

  auto image = CreateImageFrameWithOrientation(image_data, width, height,
                                               channels, *format, orientation);

  auto mp_image = std::make_unique<MpImageInternal>();
  mp_image->image = Image(std::move(image));
  *out = mp_image.release();
  return kMpOk;
}

MP_EXPORT MpStatus MpImageCreateFromImageFrame(MpImagePtr image,
                                               MpImagePtr* out,
                                               char** error_ms) {
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

MP_EXPORT MpStatus MpImageDataUint8(const MpImagePtr image, const uint8_t** out,
                                    char** error_msg) {
  auto data = GetCachedContiguousDataAttr<uint8_t>(image);
  if (data.ok()) {
    *out = *data;
  }
  return HandleStatus(data.status(), error_msg);
}

MP_EXPORT MpStatus MpImageDataUint16(const MpImagePtr image,
                                     const uint16_t** out, char** error_msg) {
  auto data = GetCachedContiguousDataAttr<uint16_t>(image);
  if (data.ok()) {
    *out = *data;
  }
  return HandleStatus(data.status(), error_msg);
}

MP_EXPORT MpStatus MpImageDataFloat32(MpImagePtr image, const float** out,
                                      char** error_msg) {
  auto data = GetCachedContiguousDataAttr<float>(image);
  if (data.ok()) {
    *out = *data;
  }
  return HandleStatus(data.status(), error_msg);
}

MP_EXPORT MpStatus MpImageGetValueUint8(const MpImagePtr image, int* pos,
                                        int pos_size, uint8_t* out,
                                        char** error_msg) {
  auto status = ValidateDimensions(image, pos_size);
  if (!status.ok()) {
    return HandleStatus(status, error_msg);
  }

  size_t byte_depth = GetImageFrameSharedPtr(image)->ByteDepth();
  if (byte_depth != 1) {
    return HandleStatus(absl::InvalidArgumentError(absl::StrFormat(
                            "Unexpected image byte depth: %d (expected 1 for "
                            "uint8_t data)",
                            byte_depth)),
                        error_msg);
  }

  std::vector<int> pos_vec(pos, pos + pos_size);
  auto value = GetValue<uint8_t>(image, pos_vec);
  if (value.ok()) {
    *out = *value;
  }
  return HandleStatus(value.status(), error_msg);
}

MP_EXPORT MpStatus MpImageGetValueUint16(const MpImagePtr image, int* pos,
                                         int pos_size, uint16_t* out,
                                         char** error_msg) {
  auto status = ValidateDimensions(image, pos_size);
  if (!status.ok()) {
    return HandleStatus(status, error_msg);
  }

  size_t byte_depth = GetImageFrameSharedPtr(image)->ByteDepth();
  if (byte_depth != 2) {
    return HandleStatus(absl::InvalidArgumentError(absl::StrFormat(
                            "Unexpected image byte depth: %d (expected 2 for "
                            "uint16_t data)",
                            byte_depth)),
                        error_msg);
  }

  std::vector<int> pos_vec(pos, pos + pos_size);
  auto value = GetValue<uint16_t>(image, pos_vec);
  if (value.ok()) {
    *out = *value;
  }
  return HandleStatus(value.status(), error_msg);
}

MP_EXPORT MpStatus MpImageGetValueFloat32(const MpImagePtr image, int* pos,
                                          int pos_size, float* out,
                                          char** error_msg) {
  auto status = ValidateDimensions(image, pos_size);
  if (!status.ok()) {
    return HandleStatus(status, error_msg);
  }

  size_t byte_depth = GetImageFrameSharedPtr(image)->ByteDepth();
  if (byte_depth != 4) {
    return HandleStatus(absl::InvalidArgumentError(absl::StrFormat(
                            "Unexpected image byte depth: %d (expected 4 for "
                            "float data)",
                            byte_depth)),
                        error_msg);
  }

  std::vector<int> pos_vec(pos, pos + pos_size);
  auto value = GetValue<float>(image, pos_vec);
  if (value.ok()) {
    *out = *value;
  }
  return HandleStatus(value.status(), error_msg);
}

MP_EXPORT void MpImageFree(MpImagePtr image) { delete image; }

}  // extern "C"
