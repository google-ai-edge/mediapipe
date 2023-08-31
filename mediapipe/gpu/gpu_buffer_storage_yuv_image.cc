/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/gpu/gpu_buffer_storage_yuv_image.h"

#include <cmath>
#include <memory>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "libyuv/video_common.h"
#include "mediapipe/framework/formats/frame_buffer.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/yuv_image.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/util/frame_buffer/frame_buffer_util.h"

namespace mediapipe {

namespace {

// Default data alignment.
constexpr int kDefaultDataAligment = 16;

GpuBufferFormat GpuBufferFormatForFourCC(libyuv::FourCC fourcc) {
  switch (fourcc) {
    case libyuv::FOURCC_NV12:
      return GpuBufferFormat::kNV12;
    case libyuv::FOURCC_NV21:
      return GpuBufferFormat::kNV21;
    case libyuv::FOURCC_YV12:
      return GpuBufferFormat::kYV12;
    case libyuv::FOURCC_I420:
      return GpuBufferFormat::kI420;
    default:
      return GpuBufferFormat::kUnknown;
  }
}

libyuv::FourCC FourCCForGpuBufferFormat(GpuBufferFormat format) {
  switch (format) {
    case GpuBufferFormat::kNV12:
      return libyuv::FOURCC_NV12;
    case GpuBufferFormat::kNV21:
      return libyuv::FOURCC_NV21;
    case GpuBufferFormat::kYV12:
      return libyuv::FOURCC_YV12;
    case GpuBufferFormat::kI420:
      return libyuv::FOURCC_I420;
    default:
      return libyuv::FOURCC_ANY;
  }
}

FrameBuffer::Format FrameBufferFormatForFourCC(libyuv::FourCC fourcc) {
  switch (fourcc) {
    case libyuv::FOURCC_NV12:
      return FrameBuffer::Format::kNV12;
    case libyuv::FOURCC_NV21:
      return FrameBuffer::Format::kNV21;
    case libyuv::FOURCC_YV12:
      return FrameBuffer::Format::kYV12;
    case libyuv::FOURCC_I420:
      return FrameBuffer::Format::kYV21;
    default:
      return FrameBuffer::Format::kUNKNOWN;
  }
}

// Converts a YuvImage into a FrameBuffer that shares the same data buffers.
std::shared_ptr<FrameBuffer> YuvImageToFrameBuffer(
    std::shared_ptr<YUVImage> yuv_image) {
  FrameBuffer::Format format = FrameBufferFormatForFourCC(yuv_image->fourcc());
  FrameBuffer::Dimension dimension{/*width=*/yuv_image->width(),
                                   /*height=*/yuv_image->height()};
  std::vector<FrameBuffer::Plane> planes;
  ABSL_CHECK(yuv_image->mutable_data(0) != nullptr && yuv_image->stride(0) > 0)
      << "Invalid YuvImage. Expected plane at index 0 to be non-null and have "
         "stride > 0.";
  planes.emplace_back(
      yuv_image->mutable_data(0),
      FrameBuffer::Stride{/*row_stride_bytes=*/yuv_image->stride(0),
                          /*pixel_stride_bytes=*/1});
  switch (format) {
    case FrameBuffer::Format::kNV12:
    case FrameBuffer::Format::kNV21: {
      ABSL_CHECK(yuv_image->mutable_data(1) != nullptr &&
                 yuv_image->stride(1) > 0)
          << "Invalid YuvImage. Expected plane at index 1 to be non-null and "
             "have stride > 0.";
      planes.emplace_back(
          yuv_image->mutable_data(1),
          FrameBuffer::Stride{/*row_stride_bytes=*/yuv_image->stride(1),
                              /*pixel_stride_bytes=*/2});
      break;
    }
    case FrameBuffer::Format::kYV12:
    case FrameBuffer::Format::kYV21: {
      ABSL_CHECK(
          yuv_image->mutable_data(1) != nullptr && yuv_image->stride(1) > 0 &&
          yuv_image->mutable_data(2) != nullptr && yuv_image->stride(2) > 0)
          << "Invalid YuvImage. Expected planes at indices 1 and 2 to be "
             "non-null and have stride > 0.";
      planes.emplace_back(
          yuv_image->mutable_data(1),
          FrameBuffer::Stride{/*row_stride_bytes=*/yuv_image->stride(1),
                              /*pixel_stride_bytes=*/1});
      planes.emplace_back(
          yuv_image->mutable_data(2),
          FrameBuffer::Stride{/*row_stride_bytes=*/yuv_image->stride(2),
                              /*pixel_stride_bytes=*/1});
      break;
    }
    default:
      ABSL_LOG(FATAL)
          << "Invalid format. Only FOURCC_NV12, FOURCC_NV21, FOURCC_YV12 and "
             "FOURCC_I420 are supported.";
  }
  return std::make_shared<FrameBuffer>(planes, dimension, format);
}

// Converts a YUVImage into an ImageFrame with ImageFormat::SRGB format.
// Note that this requires YUV -> RGB conversion.
std::shared_ptr<ImageFrame> YuvImageToImageFrame(
    std::shared_ptr<YUVImage> yuv_image) {
  auto yuv_buffer = YuvImageToFrameBuffer(yuv_image);
  // Allocate the RGB ImageFrame to return.
  auto image_frame = std::make_shared<ImageFrame>(
      ImageFormat::SRGB, yuv_buffer->dimension().width,
      yuv_buffer->dimension().height);
  // Wrap it into a FrameBuffer
  std::vector<FrameBuffer::Plane> planes{
      {image_frame->MutablePixelData(),
       {/*row_stride_bytes=*/image_frame->WidthStep(),
        /*pixel_stride_bytes=*/image_frame->NumberOfChannels() *
            image_frame->ChannelSize()}}};
  auto rgb_buffer =
      FrameBuffer(planes, yuv_buffer->dimension(), FrameBuffer::Format::kRGB);
  // Convert.
  ABSL_CHECK_OK(frame_buffer::Convert(*yuv_buffer, &rgb_buffer));
  return image_frame;
}

}  // namespace

GpuBufferStorageYuvImage::GpuBufferStorageYuvImage(
    std::shared_ptr<YUVImage> yuv_image) {
  ABSL_CHECK(GpuBufferFormatForFourCC(yuv_image->fourcc()) !=
             GpuBufferFormat::kUnknown)
      << "Invalid format. Only FOURCC_NV12, FOURCC_NV21, FOURCC_YV12 and "
         "FOURCC_I420 are supported.";
  yuv_image_ = yuv_image;
}

GpuBufferStorageYuvImage::GpuBufferStorageYuvImage(int width, int height,
                                                   GpuBufferFormat format) {
  libyuv::FourCC fourcc = FourCCForGpuBufferFormat(format);
  int y_stride = std::ceil(1.0f * width / kDefaultDataAligment);
  auto y_data = std::make_unique<uint8_t[]>(y_stride * height);
  switch (fourcc) {
    case libyuv::FOURCC_NV12:
    case libyuv::FOURCC_NV21: {
      // Interleaved U/V planes, 2x2 downsampling.
      int uv_width = 2 * std::ceil(0.5f * width);
      int uv_height = std::ceil(0.5f * height);
      int uv_stride = std::ceil(1.0f * uv_width / kDefaultDataAligment);
      auto uv_data = std::make_unique<uint8_t[]>(uv_stride * uv_height);
      yuv_image_ = std::make_shared<YUVImage>(
          fourcc, std::move(y_data), y_stride, std::move(uv_data), uv_stride,
          nullptr, 0, width, height);
      break;
    }
    case libyuv::FOURCC_YV12:
    case libyuv::FOURCC_I420: {
      // Non-interleaved U/V planes, 2x2 downsampling.
      int uv_width = std::ceil(0.5f * width);
      int uv_height = std::ceil(0.5f * height);
      int uv_stride = std::ceil(1.0f * uv_width / kDefaultDataAligment);
      auto u_data = std::make_unique<uint8_t[]>(uv_stride * uv_height);
      auto v_data = std::make_unique<uint8_t[]>(uv_stride * uv_height);
      yuv_image_ = std::make_shared<YUVImage>(
          fourcc, std::move(y_data), y_stride, std::move(u_data), uv_stride,
          std::move(v_data), uv_stride, width, height);
      break;
    }
    default:
      ABSL_LOG(FATAL)
          << "Invalid format. Only kNV12, kNV21, kYV12 and kYV21 are supported";
  }
}

GpuBufferFormat GpuBufferStorageYuvImage::format() const {
  return GpuBufferFormatForFourCC(yuv_image_->fourcc());
}

std::shared_ptr<const FrameBuffer> GpuBufferStorageYuvImage::GetReadView(
    internal::types<FrameBuffer>) const {
  return YuvImageToFrameBuffer(yuv_image_);
}

std::shared_ptr<FrameBuffer> GpuBufferStorageYuvImage::GetWriteView(
    internal::types<FrameBuffer>) {
  return YuvImageToFrameBuffer(yuv_image_);
}

std::shared_ptr<const ImageFrame> GpuBufferStorageYuvImage::GetReadView(
    internal::types<ImageFrame>) const {
  return YuvImageToImageFrame(yuv_image_);
}

std::shared_ptr<ImageFrame> GpuBufferStorageYuvImage::GetWriteView(
    internal::types<ImageFrame>) {
  // Not supported on purpose: writes into the resulting ImageFrame cannot
  // easily be ported back to the original YUV image.
  ABSL_LOG(FATAL) << "GetWriteView<ImageFrame> is not supported.";
}
}  // namespace mediapipe
