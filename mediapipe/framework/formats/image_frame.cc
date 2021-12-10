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
//
// Definitions for ImageFrame.

#include "mediapipe/framework/formats/image_frame.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <utility>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/port/aligned_malloc_and_free.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/proto_ns.h"

namespace mediapipe {

namespace {

int CountOnes(uint32 n) {
#if (defined(__i386__) || defined(__x86_64__)) && defined(__POPCNT__) && \
    defined(__GNUC__)
  return __builtin_popcount(n);
#else
  n -= ((n >> 1) & 0x55555555);
  n = ((n >> 2) & 0x33333333) + (n & 0x33333333);
  return static_cast<int>((((n + (n >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24);
#endif
}

}  // namespace

const ImageFrame::Deleter ImageFrame::PixelDataDeleter::kArrayDelete =
    std::default_delete<uint8[]>();
const ImageFrame::Deleter ImageFrame::PixelDataDeleter::kFree = free;
const ImageFrame::Deleter ImageFrame::PixelDataDeleter::kAlignedFree =
    aligned_free;
const ImageFrame::Deleter ImageFrame::PixelDataDeleter::kNone = [](uint8* x) {};

const uint32 ImageFrame::kDefaultAlignmentBoundary;
const uint32 ImageFrame::kGlDefaultAlignmentBoundary;

ImageFrame::ImageFrame()
    : format_(ImageFormat::UNKNOWN), width_(0), height_(0), width_step_(0) {}

ImageFrame::ImageFrame(ImageFormat::Format format, int width, int height,
                       uint32 alignment_boundary)
    : format_(format), width_(width), height_(height) {
  Reset(format, width, height, alignment_boundary);
}

ImageFrame::ImageFrame(ImageFormat::Format format, int width, int height)
    : format_(format), width_(width), height_(height) {
  Reset(format, width, height, kDefaultAlignmentBoundary);
}

ImageFrame::ImageFrame(ImageFormat::Format format, int width, int height,
                       int width_step, uint8* pixel_data,
                       ImageFrame::Deleter deleter) {
  AdoptPixelData(format, width, height, width_step, pixel_data, deleter);
}

ImageFrame::ImageFrame(ImageFrame&& move_from) { *this = std::move(move_from); }

ImageFrame& ImageFrame::operator=(ImageFrame&& move_from) {
  pixel_data_ = std::move(move_from.pixel_data_);
  format_ = move_from.format_;
  width_ = move_from.width_;
  height_ = move_from.height_;
  width_step_ = move_from.width_step_;

  move_from.format_ = ImageFormat::UNKNOWN;
  move_from.width_ = 0;
  move_from.height_ = 0;
  move_from.width_step_ = 0;
  return *this;
}

void ImageFrame::Reset(ImageFormat::Format format, int width, int height,
                       uint32 alignment_boundary) {
  format_ = format;
  width_ = width;
  height_ = height;
  CHECK_NE(ImageFormat::UNKNOWN, format_);
  CHECK(IsValidAlignmentNumber(alignment_boundary));
  width_step_ = width * NumberOfChannels() * ByteDepth();
  if (alignment_boundary == 1) {
    pixel_data_ = {new uint8[height * width_step_],
                   PixelDataDeleter::kArrayDelete};
  } else {
    // Increase width_step_ to the smallest multiple of alignment_boundary
    // which is large enough to hold all the data.  This is done by
    // twiddling bits.  alignment_boundary - 1 is a mask which sets all
    // the low order bits.
    width_step_ = ((width_step_ - 1) | (alignment_boundary - 1)) + 1;
    pixel_data_ = {reinterpret_cast<uint8*>(aligned_malloc(height * width_step_,
                                                           alignment_boundary)),
                   PixelDataDeleter::kAlignedFree};
  }
}

void ImageFrame::AdoptPixelData(ImageFormat::Format format, int width,
                                int height, int width_step, uint8* pixel_data,
                                ImageFrame::Deleter deleter) {
  format_ = format;
  width_ = width;
  height_ = height;
  width_step_ = width_step;

  CHECK_NE(ImageFormat::UNKNOWN, format_);
  CHECK_GE(width_step_, width * NumberOfChannels() * ByteDepth());

  pixel_data_ = {pixel_data, deleter};
}

std::unique_ptr<uint8[], ImageFrame::Deleter> ImageFrame::Release() {
  return std::move(pixel_data_);
}

void ImageFrame::InternalCopyFrom(int width, int height, int width_step,
                                  int channel_size, const uint8* pixel_data) {
  CHECK_EQ(width_, width);
  CHECK_EQ(height_, height);
  // row_bytes = channel_size * num_channels * width
  const int row_bytes = channel_size * NumberOfChannels() * width;
  if (width_step == 0) {
    width_step = channel_size * NumberOfChannels() * width;
  }
  // Copy the image data to image frame's pixel_data_.
  const char* src_row = reinterpret_cast<const char*>(pixel_data);
  char* dst_row = reinterpret_cast<char*>(pixel_data_.get());
  if (width_step == row_bytes && width_step_ == row_bytes) {
    memcpy(dst_row, src_row, height_ * row_bytes);
  } else {
    for (int i = height_; i > 0; --i) {
      memcpy(dst_row, src_row, row_bytes);
      src_row += width_step;
      dst_row += width_step_;
    }
  }
}

void ImageFrame::InternalCopyToBuffer(int width_step, char* buffer) const {
  // row_bytes = channel_size * num_channels * width
  const int row_bytes = ChannelSize() * NumberOfChannels() * width_;
  if (width_step == 0) {
    width_step = ChannelSize() * NumberOfChannels() * width_;
  }
  // Copy the image data to the provided buffer.
  const char* src_row = reinterpret_cast<const char*>(pixel_data_.get());
  char* dst_row = buffer;
  if (width_step == row_bytes && width_step_ == row_bytes) {
    memcpy(dst_row, src_row, height_ * row_bytes);
  } else {
    for (int i = height_; i > 0; --i) {
      memcpy(dst_row, src_row, row_bytes);
      src_row += width_step_;
      dst_row += width_step;
    }
  }
}

void ImageFrame::SetToZero() {
  if (pixel_data_) {
    std::fill_n(pixel_data_.get(), width_step_ * height_, 0);
  }
}

void ImageFrame::SetAlignmentPaddingAreas() {
  if (!pixel_data_) {
    return;
  }
  CHECK_GE(width_, 1);
  CHECK_GE(height_, 1);

  const int pixel_size = ByteDepth() * NumberOfChannels();
  const int padding_size = width_step_ - width_ * pixel_size;
  for (int row = 0; row < height_; ++row) {
    uint8* row_start = pixel_data_.get() + width_step_ * row;
    uint8* last_pixel_in_row = row_start + (width_ - 1) * pixel_size;
    uint8* padding = row_start + width_ * pixel_size;
    int padding_index = 0;
    while (padding_index + pixel_size - 1 < padding_size) {
      // Copy the entire last pixel in the row into this padding pixel.
      for (int pixel_byte_index = 0; pixel_byte_index < pixel_size;
           ++pixel_byte_index) {
        padding[padding_index] = last_pixel_in_row[pixel_byte_index];
        ++padding_index;
      }
    }
    // Zero out any remaining space which isn't large enough for an
    // entire pixel.
    while (padding_index < padding_size) {
      padding[padding_index] = 0;
      ++padding_index;
    }
  }
}

bool ImageFrame::IsContiguous() const {
  if (!pixel_data_) {
    return false;
  }
  return width_step_ == width_ * NumberOfChannels() * ByteDepth();
}

bool ImageFrame::IsAligned(uint32 alignment_boundary) const {
  CHECK(IsValidAlignmentNumber(alignment_boundary));
  if (!pixel_data_) {
    return false;
  }
  if ((reinterpret_cast<uintptr_t>(pixel_data_.get()) % alignment_boundary) !=
      0) {
    return false;
  }
  if ((width_step_ % alignment_boundary) != 0) {
    return false;
  }
  return true;
}

// static
bool ImageFrame::IsValidAlignmentNumber(uint32 alignment_boundary) {
  return CountOnes(alignment_boundary) == 1;
}

// static
std::string ImageFrame::InvalidFormatString(ImageFormat::Format format) {
#ifdef MEDIAPIPE_PROTO_LITE
  return "Invalid format.";
#else
  const proto_ns::EnumValueDescriptor* enum_value_descriptor =
      ImageFormat::Format_descriptor()->FindValueByNumber(format);
  if (enum_value_descriptor == nullptr) {
    return absl::StrCat("Format with number ", format,
                        " is not a valid format.");
  } else {
    return absl::StrCat("Format ", enum_value_descriptor->DebugString(),
                        " is not valid in this situation.");
  }
#endif
}

int ImageFrame::NumberOfChannels() const {
  return NumberOfChannelsForFormat(format_);
}

int ImageFrame::NumberOfChannelsForFormat(ImageFormat::Format format) {
  switch (format) {
    case ImageFormat::GRAY8:
      return 1;
    case ImageFormat::GRAY16:
      return 1;
    case ImageFormat::SRGB:
      return 3;
    case ImageFormat::SRGB48:
      return 3;
    case ImageFormat::SRGBA:
      return 4;
    case ImageFormat::SRGBA64:
      return 4;
    case ImageFormat::VEC32F1:
      return 1;
    case ImageFormat::VEC32F2:
      return 2;
    case ImageFormat::LAB8:
      return 3;
    case ImageFormat::SBGRA:
      return 4;
    default:
      LOG(FATAL) << InvalidFormatString(format);
  }
}

int ImageFrame::ChannelSize() const { return ChannelSizeForFormat(format_); }

int ImageFrame::ChannelSizeForFormat(ImageFormat::Format format) {
  switch (format) {
    case ImageFormat::GRAY8:
      return sizeof(uint8);
    case ImageFormat::SRGB:
      return sizeof(uint8);
    case ImageFormat::SRGBA:
      return sizeof(uint8);
    case ImageFormat::GRAY16:
      return sizeof(uint16);
    case ImageFormat::SRGB48:
      return sizeof(uint16);
    case ImageFormat::SRGBA64:
      return sizeof(uint16);
    case ImageFormat::VEC32F1:
      return sizeof(float);
    case ImageFormat::VEC32F2:
      return sizeof(float);
    case ImageFormat::LAB8:
      return sizeof(uint8);
    case ImageFormat::SBGRA:
      return sizeof(uint8);
    default:
      LOG(FATAL) << InvalidFormatString(format);
  }
}

int ImageFrame::ByteDepth() const { return ByteDepthForFormat(format_); }

int ImageFrame::ByteDepthForFormat(ImageFormat::Format format) {
  switch (format) {
    case ImageFormat::GRAY8:
      return 1;
    case ImageFormat::GRAY16:
      return 2;
    case ImageFormat::SRGB:
      return 1;
    case ImageFormat::SRGBA:
      return 1;
    case ImageFormat::SRGB48:
      return 2;
    case ImageFormat::SRGBA64:
      return 2;
    case ImageFormat::VEC32F1:
      return 4;
    case ImageFormat::VEC32F2:
      return 4;
    case ImageFormat::LAB8:
      return 1;
    case ImageFormat::SBGRA:
      return 1;
    default:
      LOG(FATAL) << InvalidFormatString(format);
  }
}

void ImageFrame::CopyFrom(const ImageFrame& image_frame,
                          uint32 alignment_boundary) {
  // Reset the current image.
  Reset(image_frame.Format(), image_frame.Width(), image_frame.Height(),
        alignment_boundary);

  CHECK_EQ(format_, image_frame.Format());
  InternalCopyFrom(image_frame.Width(), image_frame.Height(),
                   image_frame.WidthStep(), image_frame.ChannelSize(),
                   image_frame.PixelData());
}

void ImageFrame::CopyPixelData(ImageFormat::Format format, int width,
                               int height, const uint8* pixel_data,
                               uint32 alignment_boundary) {
  CopyPixelData(format, width, height, 0 /* contiguous storage */, pixel_data,
                alignment_boundary);
}

void ImageFrame::CopyPixelData(ImageFormat::Format format, int width,
                               int height, int width_step,
                               const uint8* pixel_data,
                               uint32 alignment_boundary) {
  Reset(format, width, height, alignment_boundary);
  InternalCopyFrom(width, height, width_step, ChannelSizeForFormat(format),
                   pixel_data);
}

void ImageFrame::CopyToBuffer(uint8* buffer, int buffer_size) const {
  CHECK(buffer);
  CHECK_EQ(1, ByteDepth());
  const int data_size = width_ * height_ * NumberOfChannels();
  CHECK_LE(data_size, buffer_size);
  if (IsContiguous()) {
    // The data is stored contiguously, we can just copy.
    const uint8* src = reinterpret_cast<const uint8*>(pixel_data_.get());
    std::copy_n(src, data_size, buffer);
  } else {
    InternalCopyToBuffer(0 /* contiguous storage */,
                         reinterpret_cast<char*>(buffer));
  }
}

void ImageFrame::CopyToBuffer(uint16* buffer, int buffer_size) const {
  CHECK(buffer);
  CHECK_EQ(2, ByteDepth());
  const int data_size = width_ * height_ * NumberOfChannels();
  CHECK_LE(data_size, buffer_size);
  if (IsContiguous()) {
    // The data is stored contiguously, we can just copy.
    const uint16* src = reinterpret_cast<const uint16*>(pixel_data_.get());
    std::copy_n(src, data_size, buffer);
  } else {
    InternalCopyToBuffer(0 /* contiguous storage */,
                         reinterpret_cast<char*>(buffer));
  }
}

void ImageFrame::CopyToBuffer(float* buffer, int buffer_size) const {
  CHECK(buffer);
  CHECK_EQ(4, ByteDepth());
  const int data_size = width_ * height_ * NumberOfChannels();
  CHECK_LE(data_size, buffer_size);
  if (IsContiguous()) {
    // The data is stored contiguously, we can just copy.
    const float* src = reinterpret_cast<float*>(pixel_data_.get());
    std::copy_n(src, data_size, buffer);
  } else {
    InternalCopyToBuffer(0 /* contiguous storage */,
                         reinterpret_cast<char*>(buffer));
  }
}
}  // namespace mediapipe
